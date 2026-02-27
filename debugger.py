import torch
import numpy as np

class QuickDebugger:
    """Ultra-simple debugger. Just copy-paste this entire class."""
    
    def __init__(self):
        self.checked_coords = False
        self.checked_encoding = False
    
    def check_once(self, model, data):
        """Call this in your first training iteration."""
        from warpconvnet.geometry.types.voxels import Voxels
        
        if not self.checked_coords:
            print("\n" + "="*60)
            print("CHECK - COORDINATES")
            
            # Convert to geometry
            geom = Voxels.from_dense(data)
            coords = geom.coordinate_tensor
            
            print(f"Coordinate tensor shape: {coords.shape}")
            
            # Check if it's 2D data in 3D format
            if coords.shape[-1] == 2:
                print("OK: 2D coordinates - correct for 2D attention")
            else:
                print("BAD: not genuine 2D data")

            self.checked_coords = True
            
            # Check encoding scale
            if not self.checked_encoding:
                print("\n" + "="*60)
                print("CHECK - ENCODING SCALE")
                print("="*60)
                
                found = False
                for name, module in model.named_modules():
                    if 'bottleneck' in name.lower() or 'attention' in name.lower():
                        if hasattr(module, 'to_attn'):
                            with torch.no_grad():
                                try:
                                    features, pos_enc, _, _ = module.to_attn(geom)
                                    
                                    feat_mean = features.abs().mean().item()
                                    
                                    if pos_enc is not None:
                                        pos_mean = pos_enc.abs().mean().item()
                                        ratio = pos_mean / (feat_mean + 1e-8)
                                        
                                        print(f"Module: {name}")
                                        print(f"  Feature magnitude: {feat_mean:.6f}")
                                        print(f"  Pos enc magnitude: {pos_mean:.6f}")
                                        print(f"  Ratio (pos/feat): {ratio:.6f}")
                                        
                                        if ratio > 5.0:
                                            print("\nPROBLEM FOUND:")
                                            print("   Positional encoding is MUCH larger than features")
                                            print("   This can cause it to dominate in flash attention")
                                            print("\n   FIX: Try encoding_range=0.5 or pos_enc_scale=0.1")
                                        elif ratio < 0.1:
                                            print("\nWARNING: Pos encoding is very small")
                                        else:
                                            print("   OK: Scale looks reasonable")
                                        
                                        found = True
                                        break
                                except Exception as e:
                                    print(f"Could not check module {name}: {e}")
                
                if not found:
                    print("Could not find attention module to check encoding")
                
                self.checked_encoding = True
            
            print("="*60 + "\n")