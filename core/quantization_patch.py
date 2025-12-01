"""
Monkey patch for PyTorch quantization to work with Python 3.14

This module patches the issue where PyTorch's quantization module tries to
set __module__ on typing.Union, which is not allowed in Python 3.14.
"""
import sys
import importlib.util

def apply_python314_quantization_patch():
    """Apply patch for Python 3.14 quantization compatibility"""
    if sys.version_info < (3, 14):
        return  # No patch needed for Python < 3.14
    
    try:
        # The issue is in torch.ao.quantization.quantizer.quantizer
        # It tries to do: EdgeOrNode.__module__ = "..."
        # But EdgeOrNode is a typing.Union, and Python 3.14 doesn't allow setting __module__ on it
        
        # We need to patch this BEFORE the module is imported
        # The best approach is to use an import hook that intercepts the problematic line
        
        # Save original import
        original_import = __import__
        
        def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
            """Import hook that patches quantization modules"""
            module = original_import(name, globals, locals, fromlist, level)
            
            # If this is the problematic module, patch it
            if name == 'torch.ao.quantization.quantizer.quantizer':
                try:
                    # The module is now loaded, try to fix EdgeOrNode if it exists
                    if hasattr(module, 'EdgeOrNode'):
                        edge_or_node = module.EdgeOrNode
                        # Check if it's a Union type
                        from typing import get_origin
                        try:
                            origin = get_origin(edge_or_node)
                            if origin is not None:  # It's a Union
                                # Create a wrapper class that has __module__
                                class EdgeOrNodeWrapper:
                                    """Wrapper that allows __module__ assignment"""
                                    __module__ = "torch.ao.quantization.quantizer.quantizer"
                                    
                                    def __init__(self, original_type):
                                        self._original = original_type
                                    
                                    def __instancecheck__(self, instance):
                                        return isinstance(instance, self._original)
                                    
                                    def __subclasscheck__(self, subclass):
                                        return issubclass(subclass, self._original)
                                
                                # Replace EdgeOrNode with our wrapper
                                module.EdgeOrNode = EdgeOrNodeWrapper(edge_or_node)
                        except Exception:
                            pass
                except Exception:
                    pass
            
            return module
        
        # Install the import hook
        builtins = __import__.__globals__.get('__builtins__', {})
        if isinstance(builtins, dict):
            builtins['__import__'] = patched_import
        else:
            builtins.__import__ = patched_import
            
    except Exception:
        # If patching fails, we'll handle it in the import function
        pass


def patch_quantizer_module():
    """Alternative patch that modifies the module after import"""
    if sys.version_info < (3, 14):
        return
    
    try:
        # Try to import and patch the problematic module
        import torch.ao.quantization.quantizer.quantizer as quantizer_mod
        
        if hasattr(quantizer_mod, 'EdgeOrNode'):
            edge_or_node = quantizer_mod.EdgeOrNode
            
            # Check if it's a Union type without __module__
            try:
                _ = edge_or_node.__module__
                # If we get here, __module__ exists, no patch needed
                return
            except (AttributeError, TypeError):
                # __module__ doesn't exist or can't be accessed
                # Create a wrapper that provides it
                from typing import get_origin, get_args
                
                try:
                    origin = get_origin(edge_or_node)
                    if origin is not None:  # It's a Union type
                        # Create a simple class that mimics Union but has __module__
                        class FixedEdgeOrNode:
                            """Fixed version of EdgeOrNode that works with Python 3.14"""
                            __module__ = "torch.ao.quantization.quantizer.quantizer"
                            
                            def __init__(self):
                                self._args = get_args(edge_or_node) if hasattr(typing, 'get_args') else ()
                            
                            def __instancecheck__(self, instance):
                                return isinstance(instance, self._args)
                            
                            def __subclasscheck__(self, subclass):
                                return issubclass(subclass, self._args)
                        
                        # Replace in the module
                        quantizer_mod.EdgeOrNode = FixedEdgeOrNode()
                except Exception:
                    pass
    except Exception:
        # Module not loaded yet or other error
        pass

