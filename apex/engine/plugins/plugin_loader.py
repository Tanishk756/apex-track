"""
Plugin Loader
=============
Discovers, validates, and instantiates plugins.

Discovery flow:
1. Scan all directories in PLUGIN_SEARCH_PATHS for plugin.yaml manifests
2. Parse manifest → PluginMetadata
3. Check required capabilities against HWProfile
4. If plugin.metadata.is_agpl == True → display license notice + require confirmation
5. Import entry_point → instantiate → call plugin.load(config, hw_profile)
6. Register in PluginRegistry

License gate (AGPL):
    When an AGPL-3.0 plugin is loaded, the loader prints a notice and requires:
    - In interactive mode: CLI prompt (y/n)
    - In headless mode: env var APEX_ACCEPT_AGPL=1 or config.accept_agpl_plugins=true
    This is a legal protection, not a technical limitation.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Optional

import structlog
import yaml

from apex.engine.hal.hw_profile import Capability, HWProfile
from apex.engine.plugins.plugin_base import (
    PluginBase,
    PluginMetadata,
    PluginStatus,
    PluginType,
)

log = structlog.get_logger(__name__)

# Default search paths
_DEFAULT_PLUGIN_DIRS = [
    Path(__file__).parent.parent.parent.parent / "plugins",  # repo root /plugins
]

_AGPL_NOTICE = """
╔══════════════════════════════════════════════════════════════════════╗
║                        LICENSE NOTICE                                ║
║                                                                      ║
║  The plugin '{name}' is licensed under AGPL-3.0.                    ║
║                                                                      ║
║  By loading this plugin, you acknowledge that if you distribute      ║
║  or deploy this software as a network service, you may be required   ║
║  to release the complete source code of the combined work under      ║
║  AGPL-3.0 terms.                                                     ║
║                                                                      ║
║  For commercial use without AGPL obligations, contact Ultralytics    ║
║  (or the respective license holder) for a commercial license.        ║
║                                                                      ║
║  Set APEX_ACCEPT_AGPL=1 or accept_agpl_plugins: true in config       ║
║  to suppress this prompt in headless/automated environments.         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

_PLUGIN_TYPE_MAP: dict[str, PluginType] = {
    "detector":  PluginType.DETECTOR,
    "tracker":   PluginType.TRACKER,
    "camera":    PluginType.CAMERA,
    "telemetry": PluginType.TELEMETRY,
    "gimbal":    PluginType.GIMBAL,
    "recording": PluginType.RECORDING,
    "mission":   PluginType.MISSION,
    "generic":   PluginType.GENERIC,
}


def _parse_manifest(manifest_path: Path) -> Optional[PluginMetadata]:
    """Parse a plugin.yaml manifest into PluginMetadata. Returns None on error."""
    try:
        data = yaml.safe_load(manifest_path.read_text())
        license_str = data.get("license", "unknown")
        is_agpl = "agpl" in license_str.lower()
        return PluginMetadata(
            name=data["name"],
            version=str(data.get("version", "0.0.0")),
            plugin_type=_PLUGIN_TYPE_MAP.get(data.get("type", "generic"), PluginType.GENERIC),
            license=license_str,
            author=data.get("author", ""),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            requires_capabilities=data.get("requires_capabilities", []),
            homepage=data.get("homepage", ""),
            is_agpl=is_agpl,
        )
    except Exception as exc:
        log.warning("manifest_parse_error", path=str(manifest_path), error=str(exc))
        return None


def _check_capabilities(meta: PluginMetadata, hw: HWProfile) -> tuple[bool, list[str]]:
    """
    Verify all required capabilities are present on the HWProfile.
    Returns (ok, missing_caps).
    """
    missing = []
    for cap_name in meta.requires_capabilities:
        try:
            cap = Capability[cap_name.upper()]
            if not hw.has(cap):
                missing.append(cap_name)
        except KeyError:
            log.warning("unknown_capability", cap=cap_name, plugin=meta.name)
    return len(missing) == 0, missing


def _accept_agpl(plugin_name: str, accept_agpl: bool, interactive: bool) -> bool:
    """Return True if AGPL license is accepted."""
    # Check env var first
    if os.environ.get("APEX_ACCEPT_AGPL", "").strip() in ("1", "true", "yes"):
        return True
    if accept_agpl:
        return True
    if interactive:
        print(_AGPL_NOTICE.format(name=plugin_name))
        answer = input("Accept AGPL-3.0 license and load this plugin? [y/N]: ").strip().lower()
        return answer in ("y", "yes")
    log.error("agpl_plugin_rejected", plugin=plugin_name,
              hint="Set APEX_ACCEPT_AGPL=1 or accept_agpl_plugins: true in config")
    return False


def _import_class(entry_point: str) -> type[PluginBase]:
    """Import 'package.module.ClassName' and return the class."""
    module_path, class_name = entry_point.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not issubclass(cls, PluginBase):
        raise TypeError(f"{entry_point} does not subclass PluginBase")
    return cls


class PluginLoader:
    """
    Discovers and loads plugins from the filesystem.

    Usage::
        loader = PluginLoader(hw_profile=hw)
        plugins = await loader.load_from_dirs([Path("plugins/")])
        plugin  = await loader.load_by_name("bytetrack", config={})
    """

    def __init__(
        self,
        hw_profile: HWProfile,
        accept_agpl: bool = False,
        interactive: bool = True,
        extra_search_paths: Optional[list[Path]] = None,
    ) -> None:
        self._hw = hw_profile
        self._accept_agpl = accept_agpl
        self._interactive = interactive
        self._search_paths: list[Path] = list(_DEFAULT_PLUGIN_DIRS)
        if extra_search_paths:
            self._search_paths.extend(extra_search_paths)

        # Discovered manifests: name → (manifest_path, entry_point_str)
        self._manifest_index: dict[str, tuple[Path, str, PluginMetadata]] = {}

    def discover(self) -> dict[str, PluginMetadata]:
        """
        Scan all search paths for plugin.yaml manifests.
        Returns dict of {plugin_name: PluginMetadata}.
        """
        self._manifest_index.clear()
        found: dict[str, PluginMetadata] = {}

        for base in self._search_paths:
            if not base.exists():
                continue
            for manifest_path in base.rglob("plugin.yaml"):
                meta = _parse_manifest(manifest_path)
                if meta is None:
                    continue
                # Read entry_point from manifest
                try:
                    raw = yaml.safe_load(manifest_path.read_text())
                    entry_point = raw.get("entry_point", "")
                    if not entry_point:
                        log.warning("missing_entry_point", manifest=str(manifest_path))
                        continue
                    self._manifest_index[meta.name] = (manifest_path, entry_point, meta)
                    found[meta.name] = meta
                    log.debug("plugin_discovered", name=meta.name, license=meta.license)
                except Exception as exc:
                    log.warning("plugin_discovery_error", path=str(manifest_path), error=str(exc))

        log.info("plugin_discovery_complete", count=len(found))
        return found

    async def load(
        self, plugin_name: str, config: dict
    ) -> Optional[PluginBase]:
        """
        Load a named plugin by name. Runs capability checks and license gate.
        Returns the loaded PluginBase instance, or None on failure.
        """
        if plugin_name not in self._manifest_index:
            # Try discover first
            self.discover()
        if plugin_name not in self._manifest_index:
            log.error("plugin_not_found", name=plugin_name)
            return None

        _, entry_point, meta = self._manifest_index[plugin_name]

        # Capability check
        ok, missing = _check_capabilities(meta, self._hw)
        if not ok:
            log.error(
                "plugin_capability_missing",
                plugin=plugin_name,
                missing=missing,
                hw=self._hw.profile_name,
            )
            return None

        # AGPL license gate
        if meta.is_agpl:
            if not _accept_agpl(plugin_name, self._accept_agpl, self._interactive):
                log.warning("plugin_load_aborted_agpl", plugin=plugin_name)
                return None

        # Import and instantiate
        try:
            cls = _import_class(entry_point)
            instance: PluginBase = cls()
            instance._set_status(PluginStatus.LOADING)
            await instance.load(config, self._hw)
            instance._set_status(PluginStatus.ACTIVE)
            log.info("plugin_loaded", name=plugin_name, type=meta.plugin_type.name)
            return instance
        except Exception as exc:
            log.error("plugin_load_failed", plugin=plugin_name, error=str(exc))
            return None

    async def load_all(
        self, plugin_configs: dict[str, dict]
    ) -> dict[str, PluginBase]:
        """
        Load multiple plugins from a config dict:
            {"bytetrack": {...config...}, "rtdetr": {...config...}}
        Returns only successfully loaded plugins.
        """
        self.discover()
        results: dict[str, PluginBase] = {}
        for name, cfg in plugin_configs.items():
            plugin = await self.load(name, cfg)
            if plugin is not None:
                results[name] = plugin
        return results
