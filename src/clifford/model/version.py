from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, field
import hashlib
import json

from clifford.model.network import Network
from clifford.core.exceptions import ModelNotFoundError


@dataclass
class ModelVersion:
    version: str
    created_at: str
    architecture: str
    params: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_version: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""


class VersionManager:
    def __init__(self):
        self.versions: Dict[str, ModelVersion] = {}
        self.current_version: Optional[str] = None

    def create_version(self, network: Network, metadata: Optional[Dict[str, Any]] = None, parent_version: Optional[str] = None, tags: Optional[List[str]] = None, notes: str = "") -> str:
        version = self._generate_version(network)
        
        model_version = ModelVersion(
            version=version,
            created_at=datetime.now().isoformat(),
            architecture=network.get_architecture(),
            params=network.get_params(),
            metadata=metadata or {},
            parent_version=parent_version,
            tags=tags or [],
            notes=notes
        )
        
        self.versions[version] = model_version
        self.current_version = version
        return version

    def get_version(self, version: str) -> Optional[ModelVersion]:
        return self.versions.get(version)

    def list_versions(self) -> List[ModelVersion]:
        return list(self.versions.values())

    def delete_version(self, version: str) -> None:
        if version not in self.versions:
            raise ModelNotFoundError(f"Version {version} not found")
        del self.versions[version]
        if self.current_version == version:
            self.current_version = None

    def get_version_history(self, version: str) -> List[ModelVersion]:
        history = []
        current = version
        while current:
            model_version = self.versions.get(current)
            if model_version:
                history.append(model_version)
                current = model_version.parent_version
            else:
                break
        return history

    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        v1 = self.versions.get(version1)
        v2 = self.versions.get(version2)
        
        if not v1 or not v2:
            raise ModelNotFoundError("One or both versions not found")
        
        return {
            "architecture_same": v1.architecture == v2.architecture,
            "params_same": v1.params == v2.params,
            "version1_created": v1.created_at,
            "version2_created": v2.created_at,
            "version1_tags": v1.tags,
            "version2_tags": v2.tags
        }

    def tag_version(self, version: str, tag: str) -> None:
        if version not in self.versions:
            raise ModelNotFoundError(f"Version {version} not found")
        if tag not in self.versions[version].tags:
            self.versions[version].tags.append(tag)

    def untag_version(self, version: str, tag: str) -> None:
        if version not in self.versions:
            raise ModelNotFoundError(f"Version {version} not found")
        if tag in self.versions[version].tags:
            self.versions[version].tags.remove(tag)

    def get_versions_by_tag(self, tag: str) -> List[ModelVersion]:
        return [v for v in self.versions.values() if tag in v.tags]

    def _generate_version(self, network: Network) -> str:
        params = network.get_params()
        params_str = json.dumps(params, sort_keys=True, default=str)
        hash_obj = hashlib.sha256(params_str.encode())
        return hash_obj.hexdigest()[:12]
