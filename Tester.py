import os
import torch
from torchinfo import summary
from torchviz import make_dot

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_model(model):
    """기본 모델 테스트"""
    print(model)
    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    try:
        output = model(input_tensor)
        print("Output shape:", output.shape)
    except Exception as e:
        print("Error during forward pass:", e)

def test_model_different_channels(model_class, input_channels, num_classes, image_size=(256, 256)):
    """다양한 입력 채널로 테스트"""
    model = model_class(input_channels=input_channels, num_classes=num_classes).to(device)
    input_tensor = torch.randn(1, input_channels, *image_size).to(device)
    try:
        output = model(input_tensor)
        print(f"Output shape with input_channels={input_channels}, num_classes={num_classes}: {output.shape}")
    except Exception as e:
        print(f"Error with input_channels={input_channels}, num_classes={num_classes}: {e}")

def count_parameters_and_model_size(model):
    """모델 파라미터 및 크기 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_all_bytes = param_size + buffer_size
    size_all_mb = size_all_bytes / (1024 ** 2)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Estimated model size: {size_all_mb:.2f} MB")
    return size_all_mb

def save_and_check_model_size(model, file_path="model.pth"):
    """모델 저장 후 실제 크기 확인"""
    torch.save(model.state_dict(), file_path)
    file_size_mb = os.path.getsize(file_path) / (1024 ** 2)
    print(f"Actual saved model size: {file_size_mb:.2f} MB")
    return file_size_mb

def model_summary(model, input_size):
    """FLOPs 및 요약"""
    print(summary(model, input_size=input_size))

def check_memory_usage():
    """메모리 사용량 확인"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"Allocated memory: {allocated:.2f} MB")
        print(f"Reserved memory: {reserved:.2f} MB")
    else:
        print("CUDA is not available.")

def quantize_and_check_size(model):
    """양자화된 모델 크기 계산"""
    model_fp16 = model.half()
    print("Calculating size for FP16 (Half Precision):")
    count_parameters_and_model_size(model_fp16)

    model_int8 = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    print("Calculating size for INT8 (Quantized):")
    save_and_check_model_size(model_int8, file_path="quantized_model.pth")

def layerwise_analysis(model):
    """레이어별 파라미터 및 메모리 사용량 분석"""
    for name, param in model.named_parameters():
        param_memory = param.nelement() * param.element_size() / (1024 ** 2)
        print(f"Layer: {name}, Parameters: {param.nelement()}, Memory: {param_memory:.2f} MB")

def test_model_with_dynamic_inputs(model, input_sizes, dtype=torch.float32):
    """
    다양한 입력 크기에서 모델 테스트 (입력 텐서 dtype 조정 포함)
    """
    for size in input_sizes:
        input_tensor = torch.randn(1, 3, *size, dtype=dtype).to(device)
        try:
            output = model(input_tensor)
            print(f"Input size: {size}, Output size: {output.shape}")
        except Exception as e:
            print(f"Error with input size {size}: {e}")

def visualize_model_safe(model, input_tensor, model_class_name, dtype=torch.float32):
    """
    모델 시각화를 안전하게 수행 (입력 dtype 변환)
    """
    input_tensor = input_tensor.to(dtype=dtype, device=device)
    output = model(input_tensor)
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.render(f"model_graph_{model_class_name}", format="png")
    print(f"Model graph saved as 'model_graph_{model_class_name}.png'")

def benchmark_model(model, input_size, device=device):
    """모델 벤치마킹"""
    model.to(device)
    input_tensor = torch.randn(*input_size).to(device)
    input_tensor = input_tensor.to(dtype=next(model.parameters()).dtype)
    model.eval()

    with torch.no_grad():
        import time
        start_time = time.time()
        for _ in range(100):
            model(input_tensor)
        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        print(f"Average inference time: {avg_time:.4f} seconds on {device}")

# ============================================== 
# ================== 모델 클래스=================
# ============================================== 







# ============================================== 
# ==============================================   
# ============================================== 



# ==================== 실행 ====================
if __name__ == "__main__":
# =============================================
# =========== 모델 클래스명 입력 ===============
    model_class =   
# =============================================
# =============================================

    # 테스트할 입력 채널과 클래스 수의 리스트 정의
    input_channels_list = [1, 3, 4]  # 예: 1 (그레이스케일), 3 (RGB), 4 (RGBA)
    num_classes_list = [1, 2, 3, 4, 5, 10]    # 예: 1, 2, 3, 4, 5, 10 클래스

    # 각 조합에 대해 테스트 수행
    for input_channels in input_channels_list:
        for num_classes in num_classes_list:
            print("="*50)
            print(f"Testing with input_channels={input_channels}, num_classes={num_classes}")
            print("="*50)
            try:
                # 모델 인스턴스화
                model_instance = model_class(input_channels=input_channels, num_classes=num_classes).to(device)

                # 기본 모델 테스트
                print("Testing model with standard configuration:")
                test_model(model_instance)

                # 파라미터 및 모델 크기 계산
                print("\nModel parameters and size:")
                count_parameters_and_model_size(model_instance)

                # 모델 저장 및 실제 크기 확인
                print("\nSaving and checking actual model size:")
                save_and_check_model_size(model_instance, file_path=f"model_{input_channels}ch_{num_classes}cls.pth")

                # 모델 요약
                print("\nModel summary:")
                model_summary(model_instance, (1, input_channels, 256, 256))

                # 양자화 및 크기 확인
                print("\nQuantizing and checking model size:")
                quantize_and_check_size(model_instance)

                # 레이어별 분석
                print("\nLayerwise parameter analysis:")
                layerwise_analysis(model_instance)

                # 다양한 입력 크기로 모델 테스트
                print("\nTesting model with dynamic input sizes:")
                dtype = torch.float16 if next(model_instance.parameters()).dtype == torch.float16 else torch.float32
                test_model_with_dynamic_inputs(model_instance, [(128, 128), (256, 256), (512, 512), (1024, 1024)], dtype=dtype)

                # 모델 시각화
                print("\nVisualizing model:")
                visualize_model_safe(model_instance, torch.randn(1, input_channels, 256, 256).to(device), model_class.__name__, dtype=dtype)

                # 모델 벤치마킹
                print("\nBenchmarking model:")
                benchmark_model(model_instance, (1, input_channels, 256, 256), device=device)

            except Exception as e:
                print(f"An error occurred for input_channels={input_channels}, num_classes={num_classes}: {e}")
            print("\n\n")

