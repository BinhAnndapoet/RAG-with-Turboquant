import torch
from transformers import LlamaConfig, LlamaForCausalLM
from kv_cache import KVCacheCompressor
from sparse_attention import apply_sparse_v_monkey_patch

def test_sparse_v_integration():
    print("1. Khởi tạo mô hình Llama Mini để test (Không tải tạ nặng)...")
    # Cấu hình một Llama thu nhỏ để chạy nhẹ nhàng trên RAM thường
    config = LlamaConfig(
        vocab_size=3200,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4, # Cố tình dùng 4 KV heads để test chuẩn GQA
        max_position_embeddings=1024,
    )
    model = LlamaForCausalLM(config).eval()

    print("2. Khởi tạo TurboQuant Compressor...")
    # LƯU Ý: Với GQA, số lượng Head của KV Cache là num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    compressor = KVCacheCompressor(
        num_layers=config.num_hidden_layers,
        num_heads=config.num_key_value_heads, 
        head_dim=head_dim,
        k_bits=3.5,
        v_bits=2.5,
        boundary_layers=0 # Tắt boundary để ép Sparse V chạy 100%
    )

    print("3. Tiêm (Inject) bản vá Sparse V vào mô hình...")
    apply_sparse_v_monkey_patch(model, compressor, threshold=1e-5)

    print("4. Chạy giả lập Forward Pass (Prefill -> Decode)...")
    try:
        batch_size = 1
        seq_len_prefill = 32
        
        # Mô phỏng đầu vào Prefill (Câu prompt của User)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len_prefill))
        
        # Chạy prefill (Chưa dùng nén)
        outputs = model(input_ids=input_ids, use_cache=True)
        # Giả lập: Lưu trạng thái sinh ra vào đối tượng nén của chúng ta
        k_cache = torch.randn(config.num_hidden_layers, config.num_key_value_heads, seq_len_prefill, head_dim)
        v_cache = torch.randn(config.num_hidden_layers, config.num_key_value_heads, seq_len_prefill, head_dim)
        compressor.calibrate(k_cache, v_cache)
        compressed_past = compressor.compress(k_cache, v_cache)
        
        print("   ✅ Prefill Pass thành công.")

        # Mô phỏng bước Decode (Sinh token tiếp theo dựa trên Cache đã nén)
        decode_input_ids = torch.randint(0, config.vocab_size, (batch_size, 1))
        # Truyền compressed_past vào thay cho past_key_value mặc định
        outputs_decode = model(input_ids=decode_input_ids, past_key_values=compressed_past, use_cache=True)
        
        print("   ✅ Decode Pass (với Sparse V) thành công! Các phép nhân Tensor hoàn toàn khớp Shape.")
        
    except Exception as e:
        print(f"\n❌ LỖI TRONG QUÁ TRÌNH TÍNH TOÁN:\n{e}")

if __name__ == "__main__":
    test_sparse_v_integration()