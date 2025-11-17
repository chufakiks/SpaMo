import torch
import os
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import random


class DatasetLoader(torch.utils.data.Dataset):

    def __init__(
        self,
        anno_root: str,
        vid_root: str,
        feat_root: str,
        mae_feat_root: str,
        mode: str = 'train',
        spatial: bool = False,
        spatiotemporal: bool = False,
        spatial_postfix: str = '',
        spatiotemporal_postfix: Union[str, List[str]] = '',
        anno_filename: str = '{mode}_info.npy', 
        default_lang: str = 'Unknown',            
        lang_variants: List[str] = None,          
        video_path_format: Optional[str] = None,  
        required_fields: List[str] = None         
    ):

    super().__init__()

    """
        Initialize the dataset.
        
        Args:
            anno_root: Root directory for annotation files
            vid_root: Root directory for video files
            feat_root: Root directory for spatial features
            mae_feat_root: Root directory for spatiotemporal features
            mode: Dataset split (e.g., 'train', 'dev', 'test', 'val')
            spatial: Whether to load spatial features
            spatiotemporal: Whether to load spatiotemporal features
            spatial_postfix: Filename postfix for spatial features
            spatiotemporal_postfix: Filename postfix for spatiotemporal features
            anno_filename: Annotation filename template (e.g., '{mode}_info.npy', '{mode}_info_ml.npy')
            default_lang: Default language code for dataset
            lang_variants: List of language variants to look for (e.g., ['en', 'es', 'fr'])
            video_path_format: Optional custom video path format string
            required_fields: List of required fields in annotation dict (e.g., ['fileid', 'text', 'gloss'])
        """
        
        self.anno_root = Path(anno_root)
        self.vid_root = Path(vid_root)
        self.feat_root = Path(feat_root)
        self.mae_feat_root = Path(mae_feat_root)
        self.mode = mode
        self.spatial = spatial
        self.spatiotemporal = spatiotemporal
        self.spatial_postfix = spatial_postfix
        self.spatiotemporal_postfix = spatiotemporal_postfix
        self.default_lang = default_lang
        self.lang_variants = lang_variants or []
        self.video_path_format = video_path_format
        self.required_fields = required_fields or ['fileid', 'text', 'gloss', 'folder']
        
        # Validate inputs
        if not (spatial or spatiotemporal):
            raise ValueError("At least one of 'spatial' or 'spatiotemporal' must be True")
        
        # Load annotations
        anno_filename = anno_filename.format(mode=mode)  # Replace {mode} placeholder
        anno_path = self.anno_root / anno_filename
        if not anno_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")
        
        self.data = np.load(anno_path, allow_pickle=True).item()
        
        # Validate annotation format
        self._validate_annotations()
        
        # Set up directory paths
        self.spatial_dir = self.feat_root / self.mode
        self.spatiotemporal_dir = self.mae_feat_root / self.mode
        
        # Validate that key directories exist
        self._validate_directories()

    def _validate_annotations(self) -> None:
        """Validate that annotations have required fields."""
        if len(self.data) == 0:
            raise ValueError("Annotation file is empty")
        
        # Check first item for required fields
        first_item = self.data[0]
        missing_fields = [f for f in self.required_fields if f not in first_item]
        if missing_fields:
            raise ValueError(f"Missing required fields in annotations: {missing_fields}")

    def _validate_directories(self) -> None:
        """Validate that all necessary directories exist."""
        if self.spatial and not self.spatial_dir.exists():
            raise FileNotFoundError(f"Spatial feature directory not found: {self.spatial_dir}")
        
        if self.spatiotemporal and not self.spatiotemporal_dir.exists():
            raise FileNotFoundError(f"Spatiotemporal feature directory not found: {self.spatiotemporal_dir}")

    def _load_spatial_features(self, file_id: str) -> torch.Tensor:
        """Load spatial features for a given file ID."""
        feat_path = self.spatial_dir / f"{file_id}{self.spatial_postfix}.npy"
        if not feat_path.exists():
            raise FileNotFoundError(f"Spatial feature file not found: {feat_path}")
        
        return torch.tensor(np.load(feat_path))

    def _load_spatiotemporal_features(self, file_id: str) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Load spatiotemporal features for a given file ID."""
        if isinstance(self.spatiotemporal_postfix, str):
            glor_path = self.spatiotemporal_dir / f"{file_id}{self.spatiotemporal_postfix}.npy"
            if not glor_path.exists():
                raise FileNotFoundError(f"Spatiotemporal feature file not found: {glor_path}")
            return torch.tensor(np.load(glor_path))
        else:
            # Handle multiple spatiotemporal features
            features = []
            for postfix in self.spatiotemporal_postfix:
                path = self.spatiotemporal_dir / f"{file_id}{postfix}.npy"
                if not path.exists():
                    raise FileNotFoundError(f"Spatiotemporal feature file not found: {path}")
                features.append(torch.tensor(np.load(path)))
            return features

    def _build_video_path(self, data: Dict) -> str:
        """Build video path using configured format or default."""
        if self.video_path_format:
            # Use custom format string
            try:
                return self.video_path_format.format(**data)
            except KeyError as e:
                raise ValueError(f"Video path format references missing field: {e}")
        else:
            # Default: just return folder if it exists
            if 'folder' in data:
                return str(self.vid_root / data['folder'])
            else:
                return ''

    def _get_text_field(self, data: Dict, lang: str = 'text') -> str:
        """Get text field, handling multiple language variants."""
        field_name = f'{lang}_text' if lang != 'text' else 'text'
        return data.get(field_name, '')

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a dataset item by index."""
        data = self.data[index]
        file_id = data['fileid']
        pixel_value = None
        glor_value = None
        
        # Load spatial features if enabled
        if self.spatial:
            try:
                pixel_value = self._load_spatial_features(file_id)
            except FileNotFoundError as e:
                print(f"Warning: {e}. Returning empty tensor.")
                pixel_value = torch.tensor([])
        
        # Load spatiotemporal features if enabled
        if self.spatiotemporal:
            try:
                glor_value = self._load_spatiotemporal_features(file_id)
            except FileNotFoundError as e:
                print(f"Warning: {e}. Returning empty tensor.")
                if isinstance(self.spatiotemporal_postfix, str):
                    glor_value = torch.tensor([])
                else:
                    glor_value = [torch.tensor([])]
        
        # Create result dictionary
        result = {
            'pixel_value': pixel_value,
            'glor_value': glor_value,
            'bool_mask_pos': None,
            'text': data.get('text', '').strip() if 'text' in data else '',
            'gloss': data.get('gloss', ''),
            'id': file_id,
            'num_frames': len(pixel_value) if pixel_value is not None and len(pixel_value) > 0 else 0,
            'vid_path': self._build_video_path(data),
            'lang': self.default_lang
        }
        
        # Add language variants if configured
        for lang in self.lang_variants:
            result[f'{lang}_text'] = self._get_text_field(data, lang)
        
        # Store original data for reference
        result['original_info'] = data
        
        return result

    def __len__(self) -> int:
        """Get the number of items in the dataset."""
        return len(self.data) - 1

    @staticmethod
    def collate_fn(batch: List[Dict]) -> List[Dict]:
        """Collate function for batch processing."""
        return batch
