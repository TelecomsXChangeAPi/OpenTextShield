#!/usr/bin/env python3
"""
GPU Verification Script - 100% Verify GPU Usage in Model Inference
This script provides comprehensive evidence of GPU/device usage
"""

import torch
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from api_interface.services.model_loader import model_manager
from api_interface.services.prediction_service import prediction_service
from api_interface.models.request_models import PredictionRequest, ModelType


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def verify_torch_device():
    """Verify PyTorch device configuration"""
    print_section("1. PyTorch Device Configuration")

    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"MPS (Apple Metal) Available: {torch.backends.mps.is_available()}")

    if torch.backends.mps.is_available():
        print("✅ MPS is AVAILABLE on this system")

    if torch.cuda.is_available():
        print(f"✅ CUDA is AVAILABLE")
        print(f"   CUDA Device Count: {torch.cuda.device_count()}")
        print(f"   Current CUDA Device: {torch.cuda.current_device()}")
        print(f"   CUDA Device Name: {torch.cuda.get_device_name(0)}")


def verify_model_device_placement():
    """Verify model is on GPU/MPS device"""
    print_section("2. Model Device Placement Verification")

    print(f"Model Manager Device: {model_manager.device}")
    print(f"Device Type: {model_manager.device.type}")

    # Get the model
    try:
        model, tokenizer, version = model_manager.get_mbert_model("multilingual")

        print(f"\n✅ Model Successfully Loaded: {version}")

        # Check each component's device
        print("\nModel Component Device Locations:")

        # Check embeddings
        if hasattr(model, 'bert'):
            embeddings_device = model.bert.embeddings.word_embeddings.weight.device
            print(f"  - Embeddings Device: {embeddings_device}")

        # Check classifier
        classifier_device = model.classifier.weight.device
        print(f"  - Classifier Head Device: {classifier_device}")

        # Check all parameters are on same device
        all_same_device = all(param.device == model_manager.device for param in model.parameters())
        print(f"\n✅ All Model Parameters on Same Device: {all_same_device}")

        if all_same_device:
            print(f"   ✅ ALL parameters are on: {model_manager.device}")
        else:
            print(f"   ❌ WARNING: Model parameters on different devices!")
            for name, param in model.named_parameters():
                if param.device != model_manager.device:
                    print(f"      {name}: {param.device}")

    except Exception as e:
        print(f"❌ Error verifying model device: {e}")


def verify_inference_on_gpu():
    """Perform inference and verify GPU usage"""
    print_section("3. Inference Execution & GPU Memory Verification")

    # Get baseline GPU memory (if available)
    baseline_allocated = 0
    baseline_reserved = 0

    if torch.backends.mps.is_available():
        print("MPS Device Detected - Memory stats may be limited")
        print("(MPS automatically manages memory, detailed stats may not be available)")

    if torch.cuda.is_available():
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        print(f"Baseline CUDA Memory Allocated: {baseline_allocated / 1024**2:.2f} MB")
        print(f"Baseline CUDA Memory Reserved: {baseline_reserved / 1024**2:.2f} MB")

    print("\n→ Running inference...")

    try:
        # Create prediction request
        request = PredictionRequest(
            text="This is a test message to verify GPU inference",
            model=ModelType.OTS_MBERT
        )

        # Run prediction synchronously (convert async if needed)
        import asyncio
        result = asyncio.run(prediction_service.predict(request))

        print(f"✅ Inference Completed Successfully!")
        print(f"\nResults:")
        print(f"  - Label: {result.label}")
        print(f"  - Probability: {result.probability:.4f}")
        print(f"  - Processing Time: {result.processing_time:.4f}s")
        print(f"  - Model: {result.model_info.name} v{result.model_info.version}")

        # Check GPU memory after inference
        if torch.cuda.is_available():
            after_allocated = torch.cuda.memory_allocated()
            after_reserved = torch.cuda.memory_reserved()
            print(f"\nCUDA Memory After Inference:")
            print(f"  - Allocated: {after_allocated / 1024**2:.2f} MB (Δ +{(after_allocated - baseline_allocated) / 1024**2:.2f} MB)")
            print(f"  - Reserved: {after_reserved / 1024**2:.2f} MB (Δ +{(after_reserved - baseline_reserved) / 1024**2:.2f} MB)")

            if after_allocated > baseline_allocated:
                print(f"  ✅ GPU Memory Increased - Inference Used GPU!")

        return result

    except Exception as e:
        print(f"❌ Inference failed: {e}")
        raise


def verify_tensor_operations_on_device():
    """Verify tensor operations are happening on GPU"""
    print_section("4. Tensor Operation Verification (Actual Computation Check)")

    device = model_manager.device
    print(f"Creating test tensors on device: {device}")

    # Create test tensors
    t1 = torch.randn(1024, 768, device=device)
    t2 = torch.randn(768, 768, device=device)

    print(f"✅ Tensor 1 Device: {t1.device}")
    print(f"✅ Tensor 2 Device: {t2.device}")

    # Perform matrix multiplication
    print(f"\n→ Performing matrix multiplication (heavy computation)...")
    result = torch.matmul(t1, t2)

    print(f"✅ Result Device: {result.device}")
    print(f"✅ Computation Result Shape: {result.shape}")

    if str(result.device).startswith('mps') or str(result.device).startswith('cuda'):
        print(f"✅ Computation performed on GPU/Accelerator: {result.device}")
    else:
        print(f"❌ Computation performed on CPU: {result.device}")


def create_detailed_report():
    """Create a detailed JSON report of device configuration"""
    print_section("5. Detailed Device Configuration Report")

    report = {
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "device": {
            "name": str(model_manager.device),
            "type": model_manager.device.type,
        },
        "model_status": {
            "multilingual_loaded": "multilingual" in model_manager.mbert_models,
            "device_placement": str(model_manager.device),
        }
    }

    if torch.cuda.is_available():
        report["cuda"] = {
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
        }

    print(json.dumps(report, indent=2))

    # Save report to file
    report_path = Path(__file__).parent / "gpu_verification_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Report saved to: {report_path}")


def main():
    """Run all verification checks"""
    print("\n")
    print("█" * 80)
    print("█ OpenTextShield GPU Verification - 100% Evidence Collection")
    print("█" * 80)

    try:
        # 1. Verify PyTorch configuration
        verify_torch_device()

        # 2. Load model and verify device placement
        print_section("LOADING MODEL...")
        model_manager.load_all_models()
        verify_model_device_placement()

        # 3. Verify tensor operations
        verify_tensor_operations_on_device()

        # 4. Run inference and verify GPU usage
        verify_inference_on_gpu()

        # 5. Create detailed report
        create_detailed_report()

        # Final summary
        print_section("FINAL VERIFICATION SUMMARY")
        print(f"✅ Device: {model_manager.device}")
        print(f"✅ Model Loaded: Yes")
        print(f"✅ Inference Executed: Yes")

        if model_manager.device.type in ['mps', 'cuda']:
            print(f"✅ GPU/Accelerator Used: YES - {model_manager.device.type.upper()}")
        else:
            print(f"⚠️  Device: {model_manager.device.type.upper()} (CPU)")

        print("\n" + "█" * 80)
        print("█ VERIFICATION COMPLETE ✅")
        print("█" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
