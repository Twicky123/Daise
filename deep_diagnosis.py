import os
import numpy as np
import tensorflow as tf

def deep_model_diagnosis(model_path):
    """
    深度診斷模型內部
    """
    print("\n" + "="*80)
    print("🔬 DEEP MODEL INTERNAL DIAGNOSIS")
    print("="*80)
    
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # =========================================================================
    # 1. 模型架構
    # =========================================================================
    print("\n[1/5] MODEL ARCHITECTURE")
    print("-"*80)
    model.summary()
    
    # =========================================================================
    # 2. 檢查所有層的權重
    # =========================================================================
    print("\n[2/5] LAYER WEIGHTS ANALYSIS")
    print("-"*80)
    
    critical_issues = []
    
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if not weights:
            continue
        
        print(f"\nLayer {i}: {layer.name} ({layer.__class__.__name__})")
        
        # 檢查 kernel (權重)
        if len(weights) > 0:
            kernel = weights[0]
            print(f"  Kernel shape: {kernel.shape}")
            print(f"    Mean:     {kernel.mean():10.6f}")
            print(f"    Std:      {kernel.std():10.6f}")
            print(f"    Min:      {kernel.min():10.6f}")
            print(f"    Max:      {kernel.max():10.6f}")
            print(f"    Abs mean: {np.abs(kernel).mean():10.6f}")
            
            # 檢查異常
            if kernel.std() < 1e-7:
                print(f"    ❌ CRITICAL: Kernel has ZERO variance!")
                critical_issues.append(f"Layer {i} ({layer.name}): Zero variance kernel")
            
            if np.abs(kernel).mean() < 1e-7:
                print(f"    ❌ CRITICAL: Kernel weights are all near ZERO!")
                critical_issues.append(f"Layer {i} ({layer.name}): Near-zero weights")
        
        # 檢查 bias
        if len(weights) > 1:
            bias = weights[1]
            print(f"  Bias shape: {bias.shape}")
            print(f"    Mean: {bias.mean():10.6f}")
            print(f"    Std:  {bias.std():10.6f}")
            print(f"    Min:  {bias.min():10.6f}")
            print(f"    Max:  {bias.max():10.6f}")
    
    # =========================================================================
    # 3. 檢查激活函數
    # =========================================================================
    print("\n[3/5] ACTIVATION FUNCTIONS")
    print("-"*80)
    
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'activation'):
            activation_name = layer.activation.__name__ if hasattr(layer.activation, '__name__') else str(layer.activation)
            print(f"Layer {i:2d} ({layer.name:20s}): {activation_name}")
            
            # 檢查 Dense 層是否缺少激活
            if isinstance(layer, tf.keras.layers.Dense):
                if 'prediction' not in layer.name and activation_name == 'linear':
                    print(f"    ⚠️  WARNING: Dense layer without activation!")
                    critical_issues.append(f"Layer {i} ({layer.name}): Missing activation")
    
    # =========================================================================
    # 4. 測試前向傳播 - 使用模型實際推論
    # =========================================================================
    print("\n[4/5] FORWARD PROPAGATION TEST")
    print("-"*80)
    
    # 創建三個非常不同的測試輸入
    test_cases = {
        'All zeros': np.zeros((1, 224, 224, 3), dtype=np.uint8),
        'All 255s': np.ones((1, 224, 224, 3), dtype=np.uint8) * 255,
        'Random': np.random.randint(0, 256, (1, 224, 224, 3), dtype=np.uint8),
    }
    
    print("\nTesting model with different inputs:")
    outputs = {}
    for name, test_input in test_cases.items():
        out = model(test_input, training=False).numpy()[0]
        outputs[name] = out
        print(f"  {name:12s}: {out}")
    
    # 檢查輸出是否相同
    output_values = list(outputs.values())
    all_same = all(np.allclose(output_values[0], out) for out in output_values[1:])
    
    if all_same:
        print(f"\n  ❌ CRITICAL: All outputs are IDENTICAL!")
        critical_issues.append("Model produces constant output for all inputs")
    else:
        max_diff = max(np.abs(output_values[i] - output_values[j]).max() 
                      for i in range(len(output_values)) 
                      for j in range(i+1, len(output_values)))
        print(f"\n  Max output difference: {max_diff:.8f}")
        
        if max_diff < 1e-6:
            critical_issues.append("Model outputs are nearly identical")
    
    # =========================================================================
    # 5. 最終輸出層特別檢查
    # =========================================================================
    print("\n[5/5] FINAL OUTPUT LAYER ANALYSIS")
    print("-"*80)
    
    final_layer = model.layers[-1]
    print(f"\nFinal layer: {final_layer.name}")
    print(f"  Type: {final_layer.__class__.__name__}")
    
    if hasattr(final_layer, 'activation'):
        print(f"  Activation: {final_layer.activation.__name__}")
    
    # 檢查最後一層的權重
    final_weights = final_layer.get_weights()
    if final_weights:
        final_kernel = final_weights[0]
        final_bias = final_weights[1] if len(final_weights) > 1 else None
        
        print(f"\n  Kernel:")
        print(f"    Shape: {final_kernel.shape}")
        print(f"    Mean:  {final_kernel.mean():10.6f}")
        print(f"    Std:   {final_kernel.std():10.6f}")
        
        if final_bias is not None:
            print(f"\n  Bias:")
            print(f"    Values: {final_bias}")
            
            # 計算 sigmoid(bias)
            sigmoid_bias = 1 / (1 + np.exp(-final_bias))
            
            print(f"\n  If model were just sigmoid(bias):")
            print(f"    sigmoid(bias) = {sigmoid_bias}")
            
            # 與實際輸出比較
            expected_output = outputs['All zeros']  # 使用全零輸入的輸出
            print(f"    Actual output = {expected_output}")
            print(f"    Difference    = {np.abs(sigmoid_bias - expected_output)}")
            
            diff = np.abs(sigmoid_bias - expected_output).max()
            if diff < 0.01:
                print(f"\n  ❌ SMOKING GUN: Output ≈ sigmoid(bias)!")
                print(f"     Difference is only {diff:.6f}")
                print(f"     Model is IGNORING all inputs!")
                critical_issues.append("Model output = sigmoid(bias) → completely ignoring inputs")
            else:
                print(f"\n  Output is NOT just sigmoid(bias) (diff={diff:.6f})")
    
    # =========================================================================
    # 總結
    # =========================================================================
    print("\n" + "="*80)
    print("CRITICAL ISSUES SUMMARY")
    print("="*80)
    
    if critical_issues:
        print(f"\nFound {len(critical_issues)} critical issues:\n")
        for i, issue in enumerate(critical_issues, 1):
            print(f"{i}. {issue}")
    else:
        print("\nNo critical structural issues found.")
        print("(But model still produces constant outputs)")
    
    print("\n" + "="*80)
    
    return critical_issues


# =========================
# Main
# =========================
if __name__ == '__main__':
    checkpoint_dir = 'checkpoints/scratch_aug/'
    
    print("\n" + "🔬"*40)
    print("DETAILED MODEL DIAGNOSIS")
    print("🔬"*40)
    
    # 檢查 Epoch 450
    model_path = os.path.join(checkpoint_dir, 'Epoch_450_model.h5')
    issues = deep_model_diagnosis(model_path)
    
    # 詳細建議
    print("\n" + "💡"*40)
    print("ROOT CAUSE ANALYSIS & RECOMMENDATIONS")
    print("💡"*40)
    
    print("\n" + "="*80)
    print("ROOT CAUSE:")
    print("="*80)
    
    has_missing_activation = any("Missing activation" in issue for issue in issues)
    outputs_constant = any("constant output" in issue.lower() for issue in issues)
    ignoring_inputs = any("ignoring inputs" in issue.lower() for issue in issues)
    
    if has_missing_activation:
        print("\n❌ CRITICAL PROBLEM: Dense layers lack activation functions")
        print("\nYour training code:")
        print("  model.add(kl.Dense(1024))  ← NO activation!")
        print("  model.add(kl.Dense(256))   ← NO activation!")
        
        print("\nThis causes the model to collapse into a single linear transformation:")
        print("  output = sigmoid(W3 @ W2 @ W1 @ input + combined_bias)")
        print("         = sigmoid(W_combined @ input + bias_combined)")
        
        print("\nDuring training, the model learned that the best 'linear' strategy")
        print("is to just output constant values (ignore input, only use bias).")
    
    if outputs_constant or ignoring_inputs:
        print("\n❌ CONFIRMED: Model produces constant outputs")
        print("\nThe model has learned to completely ignore the input and")
        print("only rely on the final bias term to produce outputs.")
    
    print("\n" + "="*80)
    print("SOLUTION:")
    print("="*80)
    
    print("\n✅ You MUST fix the training script and retrain:")
    
    print("\n1. Fix the model architecture (add activations):")
    print("```python")
    print("model.add(kl.Flatten())")
    print("model.add(kl.Dense(1024, activation='relu'))  # ← ADD activation='relu'")
    print("model.add(kl.Dropout(0.5))")
    print("model.add(kl.Dense(256, activation='relu'))   # ← ADD activation='relu'")
    print("model.add(kl.Dropout(0.3))")
    print("model.add(kl.Dense(4, activation='sigmoid'))")
    print("```")
    
    print("\n2. Fix the training function bugs:")
    print("```python")
    print("@tf.function")
    print("def train_step(x, y):  # ← Remove extra parameters")
    print("    with tf.GradientTape() as tape:")
    print("        logits = model(x, training=True)  # ← Add training=True")
    print("        loss_value = loss_fn(y, logits)")
    print("    grads = tape.gradient(loss_value, model.trainable_weights)")
    print("    optimizer.apply_gradients(zip(grads, model.trainable_weights))")
    print("    return loss_value")
    print("```")
    
    print("\n3. Monitor training closely:")
    print("   - Loss should DECREASE steadily")
    print("   - F1 score should INCREASE")
    print("   - Test model every 50 epochs with random inputs")
    print("   - Outputs should be DIFFERENT for different inputs")
    
    print("\n4. After retraining, verify the model:")
    print("```python")
    print("# Quick test")
    print("test1 = np.zeros((1,224,224,3), dtype=np.uint8)")
    print("test2 = np.ones((1,224,224,3), dtype=np.uint8)*255")
    print("out1 = model(test1).numpy()[0]")
    print("out2 = model(test2).numpy()[0]")
    print("print(f'Difference: {np.abs(out1-out2).max():.6f}')")
    print("# Should be > 0.01 if model works!")
    print("```")
    
    print("\n" + "="*80)
    print("\n⚠️  Current model is UNFIXABLE - you cannot just add activations")
    print("    to the saved model. You MUST retrain from scratch.")
    print("\n" + "="*80 + "\n")