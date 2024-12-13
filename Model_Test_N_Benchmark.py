import os
import torch
import torch.nn as nn
from torchinfo import summary
from torchviz import make_dot
import logging
from transformers import AutoTokenizer, AutoModel  # 필요한 경우

# ==================== 로깅 설정 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== 디바이스 설정 ====================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"사용 중인 디바이스: {device}")

# ==================== 입력 생성 함수 ====================
def generate_inputs(model_type, model_instance, input_channels=3, num_classes=10, image_size=(256, 256), text=""):
    """
    모델 유형에 따라 필요한 입력 데이터를 생성합니다.
    """
    if model_type == "cnn":
        input_tensor = torch.randn(1, input_channels, *image_size).to(device)
        return (input_tensor,)
    
    elif model_type == "llm":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # 예시로 GPT-2 사용
        inputs = tokenizer(text, return_tensors="pt").to(device)
        return (inputs.input_ids,)
    
    elif model_type == "multimodal":
        # 예시: CLIP 모델
        from transformers import CLIPTokenizer, CLIPTextModel
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        
        text = ["A photo of a cat"]
        text_inputs = tokenizer(text, return_tensors="pt").to(device)
        condition = text_model(**text_inputs).last_hidden_state
        input_tensor = torch.randn(1, input_channels, *image_size).to(device)
        return (input_tensor, condition)
    
    elif model_type == "stable_diffusion":
        # Stable Diffusion은 특별한 입력이 필요합니다. 예시로 텍스트 임베딩과 이미지 텐서를 생성
        from transformers import CLIPTokenizer, CLIPTextModel
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        
        text = ["A beautiful landscape"]
        text_inputs = tokenizer(text, return_tensors="pt").to(device)
        condition = text_model(**text_inputs).last_hidden_state
        time_embedding = torch.randn(1, 128).to(device)  # 예시 시간 임베딩
        input_tensor = torch.randn(1, input_channels, 256, 256).to(device)
        return (input_tensor, condition, time_embedding)
    
    else:
        raise ValueError(f"지원되지 않는 모델 유형: {model_type}")

# ==================== 테스트 함수들 ====================

def test_model(model_type, model, inputs):
    """기본 모델 테스트"""
    logger.info(f"모델 테스트 시작: {model}")
    try:
        if model_type in ["cnn", "llm"]:
            output = model(*inputs)
        elif model_type in ["multimodal", "stable_diffusion"]:
            output = model(*inputs)
        else:
            output = model(*inputs)
        
        if isinstance(output, (tuple, list)):
            for idx, out in enumerate(output):
                logger.info(f"출력 {idx} 크기: {out.shape}")
        else:
            logger.info(f"출력 크기: {output.shape}")
    except Exception as e:
        logger.error(f"순전파 중 오류 발생: {e}")

def count_parameters_and_model_size(model):
    """모델 파라미터 및 크기 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_all_bytes = param_size + buffer_size
    size_all_mb = size_all_bytes / (1024 ** 2)

    logger.info(f"전체 파라미터: {total_params}")
    logger.info(f"학습 가능한 파라미터: {trainable_params}")
    logger.info(f"예상 모델 크기: {size_all_mb:.2f} MB")
    return size_all_mb

def save_and_check_model_size(model, file_path="model.pth"):
    """모델 저장 후 실제 크기 확인"""
    torch.save(model.state_dict(), file_path)
    file_size_mb = os.path.getsize(file_path) / (1024 ** 2)
    logger.info(f"실제 저장된 모델 크기: {file_size_mb:.2f} MB")
    return file_size_mb

def model_summary(model, summary_input):
    """모델 요약 및 FLOPs 계산"""
    try:
        logger.info(summary(model, input_data=summary_input))
    except Exception as e:
        logger.error(f"모델 요약 중 오류 발생: {e}")

def check_memory_usage():
    """메모리 사용량 확인"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        logger.info(f"할당된 메모리: {allocated:.2f} MB")
        logger.info(f"예약된 메모리: {reserved:.2f} MB")
    else:
        logger.info("CUDA를 사용할 수 없습니다.")

def quantize_and_check_size(model, model_type):
    """양자화된 모델 크기 계산"""
    try:
        if model_type in ["cnn", "multimodal", "stable_diffusion"]:
            model_fp16 = model.half()
            logger.info("FP16 (Half Precision) 크기 계산 중:")
            count_parameters_and_model_size(model_fp16)

            model_int8 = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            logger.info("INT8 (Quantized) 크기 계산 중:")
            save_and_check_model_size(model_int8, file_path="quantized_model.pth")
        else:
            logger.warning(f"양자화가 지원되지 않는 모델 유형: {model_type}")
    except Exception as e:
        logger.error(f"양자화 중 오류 발생: {e}")

def layerwise_analysis(model):
    """레이어별 파라미터 및 메모리 사용량 분석"""
    try:
        for name, param in model.named_parameters():
            param_memory = param.nelement() * param.element_size() / (1024 ** 2)
            logger.info(f"레이어: {name}, 파라미터 수: {param.nelement()}, 메모리 사용량: {param_memory:.2f} MB")
    except Exception as e:
        logger.error(f"레이어별 분석 중 오류 발생: {e}")

def test_model_with_dynamic_inputs(model_type, model, input_sizes, dtype=torch.float32):
    """
    다양한 입력 크기에서 모델 테스트 (입력 텐서 dtype 조정 포함)
    """
    for size in input_sizes:
        try:
            if model_type == "cnn":
                input_tensor = torch.randn(1, model.input_channels, *size, dtype=dtype).to(device)
                inputs = (input_tensor,)
            elif model_type == "llm":
                # 예시: LLM은 시퀀스 길이가 변할 수 있습니다.
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                text = "A" * size[0]  # size[0]을 시퀀스 길이로 사용
                inputs_dict = tokenizer(text, return_tensors="pt").to(device)
                inputs = (inputs_dict.input_ids,)
            elif model_type == "multimodal":
                # 예시: 이미지 크기와 텍스트 길이를 동시에 조정
                from transformers import CLIPTokenizer, CLIPTextModel
                tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
                text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
                
                text = ["A photo of a cat"] * size[0]  # size[0]을 배치 크기로 사용
                text_inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
                condition = text_model(**text_inputs).last_hidden_state
                input_tensor = torch.randn(1, model.input_channels, *size, dtype=dtype).to(device)
                inputs = (input_tensor, condition)
            elif model_type == "stable_diffusion":
                # Stable Diffusion의 경우 이미지 크기와 텍스트 길이를 동시에 조정
                from transformers import CLIPTokenizer, CLIPTextModel
                tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
                text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
                
                text = ["A beautiful landscape"] * size[0]
                text_inputs = tokenizer(text, return_tensors="pt").to(device)
                condition = text_model(**text_inputs).last_hidden_state
                time_embedding = torch.randn(1, 128, dtype=dtype).to(device)
                input_tensor = torch.randn(1, model.input_channels, *size, dtype=dtype).to(device)
                inputs = (input_tensor, condition, time_embedding)
            else:
                input_tensor = torch.randn(1, model.input_channels, *size, dtype=dtype).to(device)
                inputs = (input_tensor,)
            
            output = model(*inputs)
            if isinstance(output, (tuple, list)):
                for idx, out in enumerate(output):
                    logger.info(f"입력 크기: {size}, 출력 {idx} 크기: {out.shape}")
            else:
                logger.info(f"입력 크기: {size}, 출력 크기: {output.shape}")
        except Exception as e:
            logger.error(f"입력 크기 {size}에서 오류 발생: {e}")

def visualize_model_safe(model, inputs, model_class_name, dtype=torch.float32):
    """
    모델 시각화를 안전하게 수행 (입력 dtype 변환)
    inputs: 모델이 요구하는 입력 튜플
    """
    try:
        converted_inputs = tuple(inp.to(dtype=dtype, device=device) for inp in inputs)
        output = model(*converted_inputs)
        dot = make_dot(output, params=dict(model.named_parameters()))
        dot.render(f"model_graph_{model_class_name}", format="png")
        logger.info(f"모델 그래프가 'model_graph_{model_class_name}.png'으로 저장되었습니다.")
    except Exception as e:
        logger.error(f"모델 시각화 중 오류 발생: {e}")

def benchmark_model(model, model_type, input_size, device=device):
    """모델 벤치마킹"""
    try:
        model.to(device)
        dtype = torch.float16 if model_type in ["cnn", "multimodal", "stable_diffusion"] else torch.float32
        
        inputs = generate_inputs(model_type, model, input_size=input_size)
        converted_inputs = tuple(inp.to(dtype=dtype) for inp in inputs)
        
        model.eval()
        with torch.no_grad():
            import time
            start_time = time.time()
            for _ in range(100):
                model(*converted_inputs)
            end_time = time.time()
            avg_time = (end_time - start_time) / 100
            logger.info(f"평균 추론 시간: {avg_time:.4f} 초 (디바이스: {device})")
    except Exception as e:
        logger.error(f"벤치마킹 중 오류 발생: {e}")



# ==================== 알고리즘 테스트 함수 ====================

def create_dummy_dataloader(input_size, num_samples=100, batch_size=16):
    """
    더미 데이터 로더를 생성합니다.
    """
    if len(input_size) == 4:
        # 이미지 데이터
        inputs = torch.randn(num_samples, *input_size)
        targets = torch.randint(0, 10, (num_samples,))
    elif len(input_size) == 2:
        # 시퀀스 데이터 (LLM)
        inputs = torch.randint(0, 50257, (num_samples, input_size[1]))  # GPT-2 vocab size
        targets = torch.randint(0, 50257, (num_samples, input_size[1]))
    else:
        raise ValueError("지원되지 않는 입력 크기 형식입니다.")
    
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def test_training_loop(model, dataloader, loss_fn, optimizer, num_epochs=1):
    """
    간단한 학습 루프를 테스트합니다.
    """
    model.train()
    for epoch in range(num_epochs):
        logger.info(f"에폭 {epoch+1}/{num_epochs}")
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(*inputs) if isinstance(inputs, tuple) else model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                logger.info(f"배치 {batch_idx}, 손실: {loss.item():.4f}")
    logger.info("학습 루프 테스트 완료.")

def test_optimizer_and_loss(model, dataloader, optimizer_class, loss_fn_class, lr=1e-3):
    """
    옵티마이저와 손실 함수의 동작을 테스트합니다.
    """
    optimizer = optimizer_class(model.parameters(), lr=lr)
    loss_fn = loss_fn_class()
    
    model.train()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(*inputs) if isinstance(inputs, tuple) else model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if batch_idx == 0:
            logger.info(f"초기 배치 손실: {loss.item():.4f}")
            break  # 초기 배치만 테스트

def test_algorithm_structure(model, model_type, input_size, optimizer_class, loss_fn_class, num_epochs=1):
    """
    알고리즘 구조를 테스트합니다.
    """
    # 더미 데이터 로더 생성
    if model_type in ["cnn", "multimodal", "stable_diffusion"]:
        if model_type == "multimodal":
            # 멀티모달의 경우 이미지와 텍스트를 동시에 다루므로 입력 크기를 조정
            image_size = input_size
            dataloader = create_dummy_dataloader(image_size, num_samples=100, batch_size=16)
        elif model_type == "stable_diffusion":
            # Stable Diffusion은 이미지와 추가 입력이 필요함
            image_size = input_size
            dataloader = create_dummy_dataloader(image_size, num_samples=100, batch_size=16)
        else:
            dataloader = create_dummy_dataloader(input_size, num_samples=100, batch_size=16)
    elif model_type == "llm":
        # LLM의 경우 시퀀스 데이터를 더미로 생성
        dataloader = create_dummy_dataloader(input_size=(1, input_size[1]), num_samples=100, batch_size=16)  # 예시 시퀀스 길이 128
    else:
        raise ValueError(f"알고리즘 테스트를 지원하지 않는 모델 유형: {model_type}")
    
    optimizer = optimizer_class(model.parameters(), lr=1e-3)
    loss_fn = loss_fn_class()
    
    test_training_loop(model, dataloader, loss_fn, optimizer, num_epochs=num_epochs)
    test_optimizer_and_loss(model, dataloader, optimizer_class, loss_fn_class)

# ============================================== 
# ================== 모델 클래스=================
# ============================================== 
# 사용자가 테스트할 모델을 정의하거나 가져옵니다.
# 예시로 몇 가지 모델 클래스를 정의합니다.

# # 예시 CNN 모델
# class SimpleCNN(nn.Module):
#     def __init__(self, input_channels=3, num_classes=10):
#         super(SimpleCNN, self).__init__()
#         self.input_channels = input_channels
#         self.conv = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
#         self.fc = nn.Linear(16 * 256 * 256, num_classes)
    
#     def forward(self, x):
#         x = self.conv(x)
#         x = torch.relu(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x

# # 예시 LLM 모델 (GPT-2)
# from transformers import GPT2LMHeadModel

# # 예시 Multimodal 모델 (CLIP)
# from transformers import CLIPModel

# # 예시 Stable Diffusion 모델
# # Stable Diffusion은 복잡한 구조을 가지므로, 여기서는 간단한 예시로 대체합니다.
# class StableDiffusionModel(nn.Module):
#     def __init__(self, input_channels=3, num_classes=10):
#         super(StableDiffusionModel, self).__init__()
#         self.input_channels = input_channels
#         self.condition_dim = 768
#         self.time_embedding = nn.Sequential(
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128)
#         )
#         self.conv = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
#         self.fc = nn.Linear(16 * 256 * 256, num_classes)
    
#     def forward(self, x, condition, time_embedding):
#         # 단순한 예시
#         x = self.conv(x)
#         x = torch.relu(x + condition.unsqueeze(-1).unsqueeze(-1))
#         x = x.view(x.size(0), -1)
#         x = self.fc(x + time_embedding)
#         return x

# ============================================== 
# ==============================================   
# ============================================== 

# ==================== 실행 ====================
if __name__ == "__main__":
    # =============================================
    # =========== 모델 클래스명 및 유형 입력 ==========
    # =============================================
    # 사용자가 테스트할 모델과 유형을 지정합니다.
    # 예시: CNN, LLM, Multimodal, Stable Diffusion
    model_definitions = [
        {"model_class":  ,"model_type": ""}
        # {"model_class": SimpleCNN, "model_type": "cnn"},
        # {"model_class": GPT2LMHeadModel, "model_type": "llm"},
        # {"model_class": CLIPModel, "model_type": "multimodal"},
        # {"model_class": StableDiffusionModel, "model_type": "stable_diffusion"}
    ]
    # =============================================
    # =============================================
    
    # 테스트할 입력 채널과 클래스 수의 리스트 정의
    test_configs = [
        {"input_channels": 3, "num_classes": 10, "image_size": (256, 256), "text": "A test input"}
        # {"input_channels": 4, "num_classes": 5, "image_size": (128, 128), "text": "Another test input"}
    ]
    
    # 알고리즘 테스트를 위한 옵티마이저 및 손실 함수 설정
    optimizer_class = torch.optim.Adam  # 예시로 Adam 사용
    loss_fn_class = nn.CrossEntropyLoss  # 예시로 CrossEntropyLoss 사용
    
    for model_def in model_definitions:
        model_class = model_def["model_class"]
        model_type = model_def["model_type"]
        
        for config in test_configs:
            input_channels = config.get("input_channels", 3)
            num_classes = config.get("num_classes", 10)
            image_size = config.get("image_size", (256, 256))
            text = config.get("text", "")
            
            logger.info("="*50)
            logger.info(f"{model_type} 모델 테스트 시작: {model_class.__name__} (입력 채널={input_channels}, 클래스 수={num_classes})")
            logger.info("="*50)
            
            try:
                # 모델 인스턴스화
                if model_type == "llm":
                    # LLM은 사전 학습된 모델을 로드
                    model_instance = model_class.from_pretrained("gpt2").to(device)
                elif model_type == "multimodal":
                    # Multimodal 모델은 사전 학습된 모델을 로드
                    model_instance = model_class.from_pretrained("openai/clip-vit-base-patch32").to(device)
                else:
                    # 일반적인 사용자 정의 모델
                    model_instance = model_class(input_channels=input_channels, num_classes=num_classes).to(device)
                
                # 기본 모델 테스트
                logger.info("기본 모델 테스트:")
                inputs = generate_inputs(model_type, model_instance, input_channels, num_classes, image_size, text)
                test_model(model_type, model_instance, inputs)
                
                # 파라미터 및 모델 크기 계산
                logger.info("\n모델 파라미터 및 크기 계산:")
                count_parameters_and_model_size(model_instance)
                
                # 모델 저장 및 실제 크기 확인
                logger.info("\n모델 저장 및 실제 크기 확인:")
                save_and_check_model_size(model_instance, file_path=f"model_{model_type}_{input_channels}ch_{num_classes}cls.pth")
                
                # 모델 요약
                logger.info("\n모델 요약:")
                model_summary(model_instance, inputs)
                
                # 양자화 및 크기 확인
                logger.info("\n모델 양자화 및 크기 확인:")
                quantize_and_check_size(model_instance, model_type)
                
                # 레이어별 분석
                logger.info("\n레이어별 파라미터 분석:")
                layerwise_analysis(model_instance)
                
                # 다양한 입력 크기로 모델 테스트
                logger.info("\n동적 입력 크기로 모델 테스트:")
                dynamic_input_sizes = [(128, 128), (256, 256), (512, 512), (1024, 1024)]
                test_model_with_dynamic_inputs(model_type, model_instance, dynamic_input_sizes)
                
                # 모델 시각화
                logger.info("\n모델 시각화:")
                visualize_inputs = generate_inputs(model_type, model_instance, input_channels, num_classes, image_size, text)
                visualize_model_safe(model_instance, visualize_inputs, model_class.__name__)
                
                # 모델 벤치마킹
                logger.info("\n모델 벤치마킹:")
                if model_type in ["cnn", "multimodal", "stable_diffusion"]:
                    benchmark_input_size = (1, input_channels, *image_size)
                elif model_type == "llm":
                    benchmark_input_size = (1, 128)  # 예시 시퀀스 길이 128
                else:
                    benchmark_input_size = (1, input_channels, *image_size)
                benchmark_model(model_instance, model_type, benchmark_input_size, device=device)
                
                # 메모리 사용량 확인
                logger.info("\n메모리 사용량 확인:")
                check_memory_usage()
                
                # 알고리즘 구조 테스트
                logger.info("\n알고리즘 구조 테스트:")
                if model_type == "llm":
                    input_size_for_algo = (1, 128)  # 예시 시퀀스 길이 128
                elif model_type in ["cnn", "multimodal", "stable_diffusion"]:
                    input_size_for_algo = image_size
                else:
                    input_size_for_algo = image_size
                test_algorithm_structure(model_instance, model_type, input_size_for_algo, optimizer_class, loss_fn_class, num_epochs=1)
                
            except Exception as e:
                logger.error(f"{model_type} 모델 {model_class.__name__} (입력 채널={input_channels}, 클래스 수={num_classes}) 테스트 중 오류 발생: {e}")
            logger.info("\n\n")
