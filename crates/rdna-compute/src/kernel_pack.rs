// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Release kernel packs: one architecture's indexed `kernels/compiled/<arch>`
//! tree built from an exact commit, plus the manifest an installer checks
//! before trusting it. `scripts/build-kernel-pack.sh` writes packs through
//! `hipfire-kernel-manifest`; `hipfire setup` and `install.ps1` install them
//! through `hipfire kernel-pack install`. Both sides use this module, and
//! per-object checks go through the runtime's own index check, so the
//! packager, the installer and the loader cannot disagree about validity.

use crate::compiler::KernelCompiler;
pub use crate::compiler::KERNEL_CACHE_ABI;
use crate::kernel_registry::{self, KernelEntry};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Manifest schema version. Bump on any field change.
pub const PACK_FORMAT: u32 = 1;
/// Manifest path inside the tarball, beside the `<arch>/` directory.
pub const MANIFEST_FILE: &str = "manifest.json";

/// Provenance and admission range of one pack. Serialized as `manifest.json`
/// inside the tarball (covered by the published SHA-256) and as a loose
/// `.manifest.json` release asset for inspection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackManifest {
    pub format: u32,
    pub tag: String,
    /// Full commit the objects were compiled from.
    pub commit: String,
    pub arch: String,
    /// Registry module count; every module has `.hsaco`, `.hash` and `.index.json`.
    pub modules: usize,
    pub kernel_cache_abi: u32,
    /// Registry extra hipcc flags (default `FeatureFlags` for `arch`).
    pub extra_flags: String,
    /// `hipcc --version` first line, as recorded in every index.
    pub toolchain_id: String,
    pub hip_version: String,
    /// `<rocm root>/.info/version` of the build toolchain.
    pub rocm_version: String,
    /// AMDGPU code-object version of every object (e.g. 6).
    pub code_object_version: u32,
    /// Oldest ROCm release the pack admits (inclusive).
    pub rocm_min: String,
    /// First ROCm release the pack no longer admits (exclusive).
    pub rocm_max_exclusive: String,
}

/// Release asset stem: `<stem>.tar.gz`, `<stem>.tar.gz.sha256`, `<stem>.manifest.json`.
pub fn asset_stem(tag: &str, arch: &str) -> String {
    format!("hipfire-kernels-{tag}-{arch}")
}

/// What a verified pack directory contains.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedPack {
    pub modules: usize,
    pub toolchain_id: String,
    pub code_object_version: u32,
}

/// Verify an installed-layout `<arch>` directory against the registry this
/// build compiles for `arch` and `extra_flags`.
///
/// Every registry module must be present with an index that matches the
/// checked-out source, recipe and object bytes; every object must share one
/// toolchain and one code-object version; and the directory must hold nothing
/// else. `local_toolchain_id` is the machine's device compiler identity, if
/// any: a pack from a different compiler build would be rejected by the
/// runtime and recompiled, so it is rejected here too.
///
/// Every symbol an index advertises must also be a kernel the compiled object
/// exports (registry symbols are a subset of the index by the runtime's own
/// index check), so an index can neither omit a supported export nor name one
/// the object lacks.
pub fn verify_dir(
    arch: &str,
    extra_flags: &str,
    dir: &Path,
    local_toolchain_id: Option<&str>,
) -> Result<VerifiedPack, String> {
    let entries = kernel_registry::entries(arch, extra_flags)
        .map_err(|error| format!("no indexed kernel registry for {arch}: {error:?}"))?;
    let verified = verify_entries(&entries, arch, extra_flags, dir, local_toolchain_id)?;
    verify_exports(&entries, arch, dir)?;
    Ok(verified)
}

/// Every indexed symbol of every registry module must be exported by the
/// module's object. All offenders are reported, not the first.
pub(crate) fn verify_exports(entries: &[KernelEntry], arch: &str, dir: &Path) -> Result<(), String> {
    let mut unexported = Vec::new();
    for entry in entries {
        let object = dir.join(format!("{}.hsaco", entry.module));
        let bytes = std::fs::read(&object).map_err(|e| format!("{}: {e}", object.display()))?;
        let exports =
            code_object_exports(&bytes).map_err(|e| format!("{}: {e}", object.display()))?;
        for symbol in KernelCompiler::indexed_symbols(dir, entry.module)? {
            if !exports.contains(&symbol) {
                unexported.push(format!("{}:{symbol}", entry.module));
            }
        }
    }
    if unexported.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{} indexed symbol(s) are not exported by the compiled {arch} objects: {}",
        unexported.len(),
        unexported.join(", ")
    ))
}

pub(crate) fn verify_entries(
    entries: &[KernelEntry],
    arch: &str,
    extra_flags: &str,
    dir: &Path,
    local_toolchain_id: Option<&str>,
) -> Result<VerifiedPack, String> {
    let mut expected = BTreeSet::new();
    let mut toolchain: Option<String> = None;
    let mut code_object: Option<u32> = None;
    for entry in entries {
        let key =
            KernelCompiler::packaging_hash_for(arch, entry.module, entry.source(), extra_flags);
        let (object, recorded) = KernelCompiler::check_indexed_object(
            dir,
            arch,
            entry.module,
            entry.source(),
            entry.symbols,
            extra_flags,
            &key,
            local_toolchain_id.unwrap_or(""),
        )?
        .ok_or_else(|| format!("{}: module missing from pack", entry.module))?;
        match &toolchain {
            None => toolchain = Some(recorded),
            Some(first) if *first != recorded => {
                return Err(format!(
                    "{}: built by `{recorded}`, other modules by `{first}`",
                    entry.module
                ))
            }
            Some(_) => {}
        }
        let bytes = std::fs::read(&object).map_err(|e| format!("{}: {e}", object.display()))?;
        let version =
            code_object_version(&bytes).map_err(|e| format!("{}: {e}", object.display()))?;
        match code_object {
            None => code_object = Some(version),
            Some(first) if first != version => {
                return Err(format!(
                    "{}: code object v{version}, other modules v{first}",
                    entry.module
                ))
            }
            Some(_) => {}
        }
        for suffix in ["hsaco", "hash", "index.json"] {
            expected.insert(format!("{}.{suffix}", entry.module));
        }
    }
    // A stray unindexed pair would be seeded into the hot cache and loaded
    // without any index check, so the directory is exactly the registry.
    for item in std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let item = item.map_err(|e| format!("{}: {e}", dir.display()))?;
        let name = item.file_name().to_string_lossy().into_owned();
        let kind = item.file_type().map_err(|e| format!("{name}: {e}"))?;
        if !kind.is_file() || !expected.contains(&name) {
            return Err(format!("{name}: not part of the {arch} registry"));
        }
    }
    Ok(VerifiedPack {
        modules: entries.len(),
        toolchain_id: toolchain.ok_or_else(|| format!("{arch}: empty registry"))?,
        code_object_version: code_object.ok_or_else(|| format!("{arch}: empty registry"))?,
    })
}

impl PackManifest {
    /// Admission checks that need no object bytes: schema, identity and the
    /// ROCm range. `local_rocm` is `<rocm root>/.info/version`, if known.
    pub fn check(&self, arch: &str, commit: &str, local_rocm: Option<&str>) -> Result<(), String> {
        if self.format != PACK_FORMAT {
            return Err(format!(
                "manifest format {} (installer reads {PACK_FORMAT})",
                self.format
            ));
        }
        if self.arch != arch {
            return Err(format!("pack is for {}, GPU is {arch}", self.arch));
        }
        if self.commit != commit {
            return Err(format!(
                "pack was built from {}, source is {commit}",
                self.commit
            ));
        }
        if self.kernel_cache_abi != KERNEL_CACHE_ABI {
            return Err(format!(
                "pack kernel cache ABI {} (runtime uses {KERNEL_CACHE_ABI})",
                self.kernel_cache_abi
            ));
        }
        let local = local_rocm.ok_or("local ROCm version unknown (no <root>/.info/version)")?;
        if !rocm_admitted(local, &self.rocm_min, &self.rocm_max_exclusive) {
            return Err(format!(
                "ROCm {local} is outside the pack's supported range [{}, {})",
                self.rocm_min, self.rocm_max_exclusive
            ));
        }
        Ok(())
    }

    /// The directory verification must agree with what the manifest claims.
    pub fn check_verified(&self, verified: &VerifiedPack) -> Result<(), String> {
        if self.modules != verified.modules
            || self.toolchain_id != verified.toolchain_id
            || self.code_object_version != verified.code_object_version
        {
            return Err(format!(
                "manifest claims {} modules/`{}`/v{}, objects are {}/`{}`/v{}",
                self.modules,
                self.toolchain_id,
                self.code_object_version,
                verified.modules,
                verified.toolchain_id,
                verified.code_object_version
            ));
        }
        Ok(())
    }
}

/// The `kernels/compiled/<arch>` directory beside the executable in
/// `exe_dir`: where the runtime loads a pack from and `kernel-pack install`
/// writes one.
pub fn pack_arch_dir(exe_dir: &Path, arch: &str) -> PathBuf {
    exe_dir.join("kernels").join("compiled").join(arch)
}

/// Sibling file `<arch_dir>.manifest.json` where `kernel-pack install`
/// records the manifest of the pack it installed.
pub fn installed_manifest_path(arch_dir: &Path) -> PathBuf {
    let name = arch_dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    arch_dir.with_file_name(format!("{name}.manifest.json"))
}

/// Manifest of the pack installed at `arch_dir`, if `kernel-pack install`
/// recorded one for this architecture. A missing, unreadable or foreign-arch
/// record is `None`: the directory then has no known provenance.
pub fn installed_manifest(arch_dir: &Path) -> Option<PackManifest> {
    let bytes = std::fs::read(installed_manifest_path(arch_dir)).ok()?;
    let manifest: PackManifest = serde_json::from_slice(&bytes).ok()?;
    (arch_dir.file_name()?.to_str()? == manifest.arch).then_some(manifest)
}

/// Numeric components of a dotted version, ignoring a non-numeric tail in each
/// component (`7.15.26333-0000000` → `[7, 15, 26333]`).
fn version_parts(version: &str) -> Option<Vec<u64>> {
    let parts = version
        .trim()
        .split('.')
        .map(|part| {
            let digits: String = part.chars().take_while(char::is_ascii_digit).collect();
            digits.parse::<u64>().ok()
        })
        .collect::<Option<Vec<_>>>()?;
    (!parts.is_empty()).then_some(parts)
}

fn compare_versions(a: &[u64], b: &[u64]) -> std::cmp::Ordering {
    let len = a.len().max(b.len());
    let at = |v: &[u64], i: usize| v.get(i).copied().unwrap_or(0);
    (0..len)
        .map(|i| at(a, i).cmp(&at(b, i)))
        .find(|order| order.is_ne())
        .unwrap_or(std::cmp::Ordering::Equal)
}

/// `min <= local < max_exclusive`, comparing dotted numeric versions.
pub fn rocm_admitted(local: &str, min: &str, max_exclusive: &str) -> bool {
    match (
        version_parts(local),
        version_parts(min),
        version_parts(max_exclusive),
    ) {
        (Some(local), Some(min), Some(max)) => {
            compare_versions(&local, &min).is_ge() && compare_versions(&local, &max).is_lt()
        }
        _ => false,
    }
}

/// Default admission range for a pack built with ROCm `version`: the build's
/// `major.minor` up to the next major release.
pub fn default_rocm_range(version: &str) -> Option<(String, String)> {
    let parts = version_parts(version)?;
    let major = parts[0];
    let minor = parts.get(1).copied().unwrap_or(0);
    Some((format!("{major}.{minor}"), format!("{}.0", major + 1)))
}

/// The amdgcn ELF of a `--genco` object: an uncompressed clang offload bundle
/// holding one amdgcn ELF, or that ELF itself.
fn device_elf(object: &[u8]) -> Result<&[u8], String> {
    const BUNDLE_MAGIC: &[u8] = b"__CLANG_OFFLOAD_BUNDLE__";
    let read_u64 = |at: usize| -> Result<usize, String> {
        let bytes = object
            .get(at..at.checked_add(8).ok_or("offload bundle offset overflow")?)
            .ok_or("truncated offload bundle")?;
        usize::try_from(u64::from_le_bytes(bytes.try_into().expect("8 bytes")))
            .map_err(|_| "offload bundle offset overflow".to_owned())
    };
    let elf = if object.starts_with(BUNDLE_MAGIC) {
        let count = read_u64(BUNDLE_MAGIC.len())?;
        let mut cursor = BUNDLE_MAGIC.len() + 8;
        let mut device = None;
        for _ in 0..count {
            let (offset, size, triple_len) = (
                read_u64(cursor)?,
                read_u64(cursor + 8)?,
                read_u64(cursor + 16)?,
            );
            let start = cursor + 24;
            let end = start
                .checked_add(triple_len)
                .ok_or("offload bundle offset overflow")?;
            let triple = object.get(start..end).ok_or("truncated offload bundle")?;
            cursor = end;
            if size > 0 && triple.windows(6).any(|window| window == b"amdgcn") {
                let stop = offset
                    .checked_add(size)
                    .ok_or("offload bundle offset overflow")?;
                device = Some(
                    object
                        .get(offset..stop)
                        .ok_or("truncated offload bundle entry")?,
                );
            }
        }
        device.ok_or("offload bundle has no amdgcn object")?
    } else {
        object
    };
    if elf.get(..4) != Some(b"\x7fELF".as_slice()) || elf.len() < 16 {
        return Err("not an ELF code object".to_owned());
    }
    Ok(elf)
}

/// AMDGPU code-object version of a `--genco` object: an uncompressed clang
/// offload bundle holding one amdgcn ELF, or that ELF itself.
pub fn code_object_version(object: &[u8]) -> Result<u32, String> {
    const ELFOSABI_AMDGPU_HSA: u8 = 64;
    let elf = device_elf(object)?;
    if elf[7] != ELFOSABI_AMDGPU_HSA {
        return Err(format!("ELF OS/ABI {} is not AMDGPU HSA", elf[7]));
    }
    // EI_ABIVERSION: 2 → code object v4, 3 → v5, 4 → v6.
    Ok(u32::from(elf[8]) + 2)
}

/// The kernels a code object exports: the global function symbols of its
/// `.dynsym`, the names `hipModuleGetFunction` resolves. (`<kernel>.kd`
/// descriptors are data objects, not kernels.) The object is the same
/// uncompressed bundle or bare ELF [`code_object_version`] accepts.
pub fn code_object_exports(object: &[u8]) -> Result<BTreeSet<String>, String> {
    const ELFCLASS64: u8 = 2;
    const ELFDATA2LSB: u8 = 1;
    const SHT_DYNSYM: u64 = 11;
    const STT_FUNC: u8 = 2;
    const STB_GLOBAL: u8 = 1;
    const SYM_SIZE: usize = 24;
    const SECTION_SIZE: usize = 64;
    let elf = device_elf(object)?;
    if elf[4] != ELFCLASS64 || elf[5] != ELFDATA2LSB {
        return Err("not a little-endian ELF64 code object".to_owned());
    }
    let field = |at: usize, len: usize| -> Result<u64, String> {
        let bytes = at
            .checked_add(len)
            .and_then(|end| elf.get(at..end))
            .ok_or("truncated ELF code object")?;
        Ok(bytes.iter().rev().fold(0u64, |acc, &b| (acc << 8) | u64::from(b)))
    };
    let offset = |at: usize, len: usize| -> Result<usize, String> {
        usize::try_from(field(at, len)?).map_err(|_| "ELF offset overflow".to_owned())
    };
    let section_table = offset(0x28, 8)?;
    let section_entry = offset(0x3a, 2)?;
    let section_count = offset(0x3c, 2)?;
    if section_entry < SECTION_SIZE {
        return Err("ELF section header entries are too small".to_owned());
    }
    let section_at = |index: usize| -> Result<usize, String> {
        index
            .checked_mul(section_entry)
            .and_then(|delta| section_table.checked_add(delta))
            .ok_or_else(|| "ELF section table overflow".to_owned())
    };
    let dynsym = (0..section_count)
        .find_map(|index| {
            let at = section_at(index).ok()?;
            (field(at + 4, 4).ok()? == SHT_DYNSYM).then_some(at)
        })
        .ok_or("ELF code object has no .dynsym")?;
    let (symbols, symbols_size) = (offset(dynsym + 24, 8)?, offset(dynsym + 32, 8)?);
    let strtab_at = section_at(offset(dynsym + 40, 4)?)?;
    let (strings, strings_size) = (offset(strtab_at + 24, 8)?, offset(strtab_at + 32, 8)?);
    let strings = strings
        .checked_add(strings_size)
        .and_then(|end| elf.get(strings..end))
        .ok_or("truncated ELF .dynstr")?;
    let mut exports = BTreeSet::new();
    for index in 0..symbols_size / SYM_SIZE {
        let at = symbols + index * SYM_SIZE;
        let info = *elf.get(at + 4).ok_or("truncated ELF .dynsym")?;
        let defined = field(at + 6, 2)? != 0;
        if defined && info & 0xf == STT_FUNC && info >> 4 == STB_GLOBAL {
            let name = strings
                .get(offset(at, 4)?..)
                .ok_or("ELF symbol name outside .dynstr")?;
            let end = name
                .iter()
                .position(|&b| b == 0)
                .ok_or("unterminated ELF symbol name")?;
            exports.insert(
                std::str::from_utf8(&name[..end])
                    .map_err(|_| "non-UTF-8 ELF symbol name".to_owned())?
                    .to_owned(),
            );
        }
    }
    Ok(exports)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rocm_range_is_half_open_and_numeric() {
        assert!(rocm_admitted("10.0.0", "10.0", "11.0"));
        assert!(rocm_admitted("10.12.1", "10.0", "11.0"));
        assert!(!rocm_admitted("11.0.0", "10.0", "11.0"));
        assert!(!rocm_admitted("9.9.9", "10.0", "11.0"));
        // Lexical comparison would admit 7.10 below 7.2.
        assert!(rocm_admitted("7.10.0", "7.2", "8.0"));
        assert!(!rocm_admitted("7.1.1", "7.2", "8.0"));
        assert!(!rocm_admitted("unknown", "7.2", "8.0"));
        assert_eq!(
            default_rocm_range("10.0.0"),
            Some(("10.0".into(), "11.0".into()))
        );
        assert_eq!(
            default_rocm_range("7.2.4"),
            Some(("7.2".into(), "8.0".into()))
        );
    }

    fn scratch(label: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "hipfire-kpack-{label}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn manifest(arch: &str) -> PackManifest {
        PackManifest {
            format: PACK_FORMAT,
            tag: "v0.4.3".into(),
            commit: "0".repeat(40),
            arch: arch.into(),
            modules: 3,
            kernel_cache_abi: KERNEL_CACHE_ABI,
            extra_flags: String::new(),
            toolchain_id: "HIP version: 7.2".into(),
            hip_version: "7.2".into(),
            rocm_version: "7.2.0".into(),
            code_object_version: 6,
            rocm_min: "7.2".into(),
            rocm_max_exclusive: "8.0".into(),
        }
    }

    #[test]
    fn installed_manifest_round_trips_through_the_sibling_file() {
        let root = scratch("roundtrip");
        let arch_dir = root.join("gfx1151");
        let expected = manifest("gfx1151");
        std::fs::write(
            installed_manifest_path(&arch_dir),
            serde_json::to_vec(&expected).unwrap(),
        )
        .unwrap();
        assert_eq!(
            installed_manifest_path(&arch_dir),
            root.join("gfx1151.manifest.json")
        );
        assert_eq!(installed_manifest(&arch_dir), Some(expected));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn installed_manifest_for_another_arch_is_ignored() {
        let root = scratch("mismatch");
        let arch_dir = root.join("gfx1151");
        std::fs::write(
            installed_manifest_path(&arch_dir),
            serde_json::to_vec(&manifest("gfx1201")).unwrap(),
        )
        .unwrap();
        assert_eq!(installed_manifest(&arch_dir), None);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn installed_manifest_missing_or_garbled_is_none() {
        let root = scratch("missing");
        let arch_dir = root.join("gfx1151");
        assert_eq!(installed_manifest(&arch_dir), None);
        std::fs::write(installed_manifest_path(&arch_dir), b"not json").unwrap();
        assert_eq!(installed_manifest(&arch_dir), None);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn pack_arch_dir_is_the_executable_neighbour_layout() {
        assert_eq!(
            pack_arch_dir(Path::new("/opt/hipfire/bin"), "gfx1151"),
            Path::new("/opt/hipfire/bin/kernels/compiled/gfx1151")
        );
    }

    /// A minimal AMDGPU HSA ELF64 whose `.dynsym` holds `(name, STT_*, STB_*)`
    /// symbols, all defined (section index 1).
    fn elf_with_dynsym(symbols: &[(&str, u8, u8)]) -> Vec<u8> {
        let mut strtab = vec![0u8];
        let mut symtab = vec![0u8; 24];
        for (name, kind, bind) in symbols {
            symtab.extend_from_slice(&(strtab.len() as u32).to_le_bytes());
            strtab.extend_from_slice(name.as_bytes());
            strtab.push(0);
            symtab.push(bind << 4 | kind);
            symtab.push(0);
            symtab.extend_from_slice(&1u16.to_le_bytes());
            symtab.extend_from_slice(&[0u8; 16]);
        }
        let symtab_at = 64;
        let strtab_at = symtab_at + symtab.len();
        let sections_at = strtab_at + strtab.len();
        let mut elf = vec![0u8; 64];
        elf[..4].copy_from_slice(b"\x7fELF");
        elf[4] = 2; // ELFCLASS64
        elf[5] = 1; // little endian
        elf[6] = 1;
        elf[7] = 64; // AMDGPU HSA
        elf[8] = 4; // code object v6
        elf[0x28..0x30].copy_from_slice(&(sections_at as u64).to_le_bytes());
        elf[0x3a..0x3c].copy_from_slice(&64u16.to_le_bytes());
        elf[0x3c..0x3e].copy_from_slice(&3u16.to_le_bytes());
        elf.extend_from_slice(&symtab);
        elf.extend_from_slice(&strtab);
        let section = |kind: u32, offset: usize, size: usize, link: u32| {
            let mut header = vec![0u8; 64];
            header[4..8].copy_from_slice(&kind.to_le_bytes());
            header[24..32].copy_from_slice(&(offset as u64).to_le_bytes());
            header[32..40].copy_from_slice(&(size as u64).to_le_bytes());
            header[40..44].copy_from_slice(&link.to_le_bytes());
            header
        };
        elf.extend(section(0, 0, 0, 0));
        elf.extend(section(11, symtab_at, symtab.len(), 2)); // SHT_DYNSYM, strings: section 2
        elf.extend(section(3, strtab_at, strtab.len(), 0)); // SHT_STRTAB
        elf
    }

    const FUNC: u8 = 2;
    const OBJECT: u8 = 1;
    const GLOBAL: u8 = 1;
    const LOCAL: u8 = 0;

    #[test]
    fn code_object_exports_are_the_global_function_kernels() {
        let elf = elf_with_dynsym(&[
            ("kernel_a", FUNC, GLOBAL),
            ("kernel_a.kd", OBJECT, GLOBAL),
            ("kernel_b", FUNC, GLOBAL),
            ("helper", FUNC, LOCAL),
        ]);
        let expected: BTreeSet<String> = ["kernel_a", "kernel_b"].map(str::to_owned).into();
        assert_eq!(code_object_exports(&elf).unwrap(), expected);
        assert_eq!(code_object_version(&elf).unwrap(), 6);

        // The same ELF inside an uncompressed offload bundle next to a host entry.
        let triples: [&[u8]; 2] = [b"host-x86_64-unknown-linux-gnu-", b"hipv4-amdgcn-amd-amdhsa--gfx1100"];
        let payload_at = 32 + triples.iter().map(|t| 24 + t.len()).sum::<usize>();
        let mut bundle = b"__CLANG_OFFLOAD_BUNDLE__".to_vec();
        bundle.extend_from_slice(&2u64.to_le_bytes());
        for (index, triple) in triples.iter().enumerate() {
            let (offset, size) = if index == 0 { (payload_at, 0) } else { (payload_at, elf.len()) };
            for value in [offset, size, triple.len()] {
                bundle.extend_from_slice(&(value as u64).to_le_bytes());
            }
            bundle.extend_from_slice(triple);
        }
        bundle.extend_from_slice(&elf);
        assert_eq!(code_object_exports(&bundle).unwrap(), expected);

        assert!(code_object_exports(b"not an elf at all, sixteen+ bytes").is_err());
        let mut no_symbols = elf.clone();
        // Retype the dynsym section header (section 1) so no `.dynsym` remains.
        let sections_at = u64::from_le_bytes(no_symbols[0x28..0x30].try_into().unwrap()) as usize;
        no_symbols[sections_at + 64 + 4] = 1;
        assert!(code_object_exports(&no_symbols).unwrap_err().contains(".dynsym"));
    }

    #[test]
    fn pack_export_check_names_every_indexed_symbol_the_object_lacks() {
        let root = scratch("exports");
        let entries = vec![KernelEntry {
            arch: "gfx1100",
            module: "alpha",
            symbols: &["alpha_a", "alpha_b", "alpha_c"],
            source: "".into(),
            flags: Vec::new(),
            scheduler_profile: None,
        }];
        let write_index = |symbols: &[&str]| {
            let index = serde_json::json!({
                "version": 1, "module": "alpha", "symbols": symbols, "arch": "gfx1100",
                "source_sha256": "", "flags": [], "scheduler_profile": "", "cache_abi": 0,
                "packaging_key": "", "object_sha256": "", "toolchain_id": "t",
                "toolchain_sha256": "",
            });
            std::fs::write(root.join("alpha.index.json"), serde_json::to_vec(&index).unwrap())
                .unwrap();
        };
        std::fs::write(
            root.join("alpha.hsaco"),
            elf_with_dynsym(&[("alpha_a", FUNC, GLOBAL), ("alpha_b", FUNC, GLOBAL)]),
        )
        .unwrap();

        write_index(&["alpha_a", "alpha_b"]);
        assert!(verify_exports(&entries, "gfx1100", &root).is_ok());
        write_index(&["alpha_a", "alpha_b", "alpha_c", "alpha_d"]);
        let error = verify_exports(&entries, "gfx1100", &root).unwrap_err();
        assert!(error.contains("2 indexed symbol(s)"), "{error}");
        assert!(error.contains("alpha:alpha_c") && error.contains("alpha:alpha_d"), "{error}");
        assert!(!error.contains("alpha:alpha_a"), "{error}");
        let _ = std::fs::remove_dir_all(root);
    }
}
