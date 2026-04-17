import torch
import types
import math

# Import các hàm tiện ích có sẵn từ module Llama của HuggingFace
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

def apply_sparse_v_monkey_patch(model, compressor, threshold=1e-6):
    """
    Ghi đè cơ chế Attention của mô hình LLM để nhúng công cụ TurboQuant.
    Đã xử lý tường minh RoPE và GQA.
    """
    for layer_idx, layer in enumerate(model.model.layers):
        attn_module = layer.self_attn
        
        # Lưu trữ trạng thái vào nội bộ module
        attn_module.layer_idx = layer_idx
        attn_module.compressor = compressor
        attn_module.sparse_threshold = threshold
        
        # Định nghĩa lại hàm Forward
        def sparse_attention_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor = None,
            position_ids: torch.LongTensor = None,
            past_key_value = None, 
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs
        ):
            bsz, q_len, _ = hidden_states.size()

            # 1. Chiếu Q, K, V
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            # 2. Định dạng lại shape thành [batch, num_heads, seq_len, head_dim]
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            # 3. Tính toán và áp dụng Rotary Embeddings (RoPE)
            # Truyền value_states và position_ids vào lớp rotary_emb của model
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # 4. Nhân bản K, V nếu dùng GQA (Grouped-Query Attention) 
            # (Đặc trưng của Llama 3 và Qwen)
            if self.num_key_value_groups > 1:
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

            # --- LUỒNG DECODING VỚI TURBOQUANT ---
            if past_key_value is not None and hasattr(past_key_value, 'k_data'):
                
                # CHẶNG 1: GIẢI NÉN KEY VÀ TÍNH ĐIỂM
                k_history = self.compressor.decompress_k_layer(past_key_value, self.layer_idx)
                
                # Cập nhật Key hiện tại vào lịch sử
                full_keys = torch.cat([k_history, key_states], dim=2)
                
                # Q @ K^T
                attn_weights = torch.matmul(query_states, full_keys.transpose(2, 3)) / math.sqrt(self.head_dim)
                
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                    
                # Softmax để lấy điểm phân phối thực tế
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                
                # CHẶNG 2: SPARSE VALUE DECOMPRESSION (ĐIỂM NHẤN CỦA BẠN)
                v_history_sparse = self.compressor.decompress_v_layer_sparse(
                    past_key_value, 
                    self.layer_idx, 
                    attn_weights, 
                    self.sparse_threshold
                )
                
                # Cập nhật Value hiện tại
                full_values = torch.cat([v_history_sparse, value_states], dim=2)
                
                # Tính Output cuối cùng
                attn_output = torch.matmul(attn_weights, full_values)
                
            else:
                # --- LUỒNG PREFILL (Xử lý đoạn Prompt đầu vào) ---
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_output = torch.matmul(attn_weights, value_states)

            # 5. Chiếu O_proj và định dạng lại shape đầu ra
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            
            return attn_output, None, past_key_value

        # Trói buộc (bind) hàm mới vào instance đang chạy
        attn_module.forward = types.MethodType(sparse_attention_forward, attn_module)
        
    print(f"✅ Đã tiêm thành công Sparse V Attention vào {len(model.model.layers)} layers (Ngưỡng chặn: {threshold})")