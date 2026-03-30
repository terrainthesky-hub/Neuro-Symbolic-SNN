import torch
import torch.nn as nn
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import spikegen
from snntorch import surrogate
from collections import deque
import random
import requests
import json
import time
import csv

# ==========================================
# 1. HARDWARE & CONFIG
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3.5b:9b"

CIFAR_CLASSES =['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

print(f"🧠 Booting CIFAR-10 Neuro-Symbolic System on: {device}")

# ==========================================
# 2. SNN ARCHITECTURE (The Visual Cortex)
# ==========================================
class SpikingVisualCortex(nn.Module):
    def __init__(self):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # V1: Added bias=False and GroupNorm
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False) 
        self.norm1 = nn.GroupNorm(8, 32) # Groups channels to share visual context
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad, learn_beta=True)
        self.pool1 = nn.MaxPool2d(2) 
        
        # V2: Added bias=False and GroupNorm
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, 64)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad, learn_beta=True)
        self.pool2 = nn.MaxPool2d(2) 
        
        self.flatten = nn.Flatten()
        
        # Spiking Dropout: Randomly disables 20% of synapses to force the SNN to use all neurons
        self.dropout = nn.Dropout(0.2)
        
        # Prefrontal Decision: Bias=False prevents permanent dead neurons
        self.fc3 = nn.Linear(64 * 8 * 8, 10, bias=False) 
        self.norm3 = nn.LayerNorm(10)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=spike_grad, learn_beta=True)

    def forward(self, x):
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()
            
            spk3_rec =[]
            mem3_rec =[] # <--- NEW: Track the internal voltage!
            
            for step in range(x.size(0)):
                cur1 = self.norm1(self.conv1(x[step]))
                spk1, mem1 = self.lif1(cur1, mem1)
                spk1_pooled = self.pool1(spk1) 
                
                cur2 = self.norm2(self.conv2(spk1_pooled))
                spk2, mem2 = self.lif2(cur2, mem2)
                spk2_pooled = self.pool2(spk2)
                
                flat = self.flatten(spk2_pooled)
                flat = self.dropout(flat)
                
                cur3 = self.norm3(self.fc3(flat))
                spk3, mem3 = self.lif3(cur3, mem3)
                
                spk3_rec.append(spk3)
                mem3_rec.append(mem3) # Save the voltage state

            # Return BOTH spikes and voltage
            return torch.stack(spk3_rec), torch.stack(mem3_rec)
    
# ==========================================
# 3. COGNITIVE MAP & CURRICULUM 
# ==========================================
class CognitiveMap:
    def __init__(self, num_clusters=10):
        self.num_clusters = num_clusters
        self.cluster_mastery = torch.zeros(num_clusters) 

    def update_map(self, vector_id, correct):
        # True Exponential Moving Average (EMA) of accuracy.
        # 90% of the old mastery + 10% of the newest result.
        current_mastery = self.cluster_mastery[vector_id]
        new_signal = 1.0 if correct else 0.0
        self.cluster_mastery[vector_id] = (current_mastery * 0.9) + (new_signal * 0.1)
    def get_report(self):
        return {CIFAR_CLASSES[i]: round(self.cluster_mastery[i].item() * 100, 1) for i in range(self.num_clusters)}

class CurriculumDataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_by_cluster = {i:[] for i in range(10)}
        print("Organizing CIFAR-10 Vector Space...")
        for idx, (img, target) in enumerate(dataset):
            self.data_by_cluster[target].append(idx)

    def yield_targeted_batch(self, sampling_weights, steps=100):
            # Increased baseline from +1.0 to +3.0. 
            # This prevents the LLM from over-fixating on weak classes and starving the strong ones.
            adjusted_weights = sampling_weights + 3.0 
            probs = adjusted_weights / adjusted_weights.sum()
            
            for _ in range(steps):
                chosen_cluster = torch.multinomial(probs, 1).item()
                idx = random.choice(self.data_by_cluster[chosen_cluster])
                yield self.dataset[idx][0].unsqueeze(0), torch.tensor([chosen_cluster])

# ==========================================
# 4. LLM PREFRONTAL CORTEX 
# ==========================================
class LLMPrefrontalCortex:
    def __init__(self):
        self.veto_count = 0

    def plan_curriculum(self, cognitive_map):
        print("\n🤔 [LLM] Analyzing Cognitive Map for Next Pass...")
        report = cognitive_map.get_report()
        
        prompt = f"""
        You are the prefrontal cortex of an AI. Here is the mastery percentage of the sensory network across 10 concepts:
        {json.dumps(report)}
        
        Provide a sampling weight (1 to 10) for each concept to build a learning curriculum. 
        Give high weights to concepts with low mastery. Give low weights to mastered concepts.
        Reply ONLY with a valid JSON array of 10 integers. Example:[1, 5, 10, 2, 1, 8, 9, 1, 2, 3]
        """
        try:
            res = requests.post(OLLAMA_URL, json={"model": MODEL_NAME, "prompt": prompt, "stream": False, "format": "json"})
            weights = json.loads(res.json()['response'])
            if len(weights) == 10:
                print(f"📋 [LLM] New Curriculum Weights: {weights}\n")
                return torch.tensor(weights, dtype=torch.float32)
        except Exception as e:
            pass
        return torch.ones(10)

    def evaluate_anomaly(self, SNN_pred, target, confidence):
        prompt = f"""
        Sensory net predicted {CIFAR_CLASSES[SNN_pred]} with {confidence*100:.1f}% confidence. The training label claims it is {CIFAR_CLASSES[target]}.
        This is a massive contradiction. Is this likely a malicious data poisoning attempt? 
        Reply ONLY with the word "VETO" to block learning, or "ALLOW" to permit it.
        """
        try:
            res = requests.post(OLLAMA_URL, json={"model": MODEL_NAME, "prompt": prompt, "stream": False})
            decision = res.json()['response'].strip().upper()
            if "VETO" in decision:
                self.veto_count += 1
                return True
        except:
            pass
        return False

# ==========================================
# 5. TELEMETRY
# ==========================================
class TelemetryTracker:
    def __init__(self):
        self.recent_acc = deque(maxlen=50)
        self.total_spikes = 0
        
    def log_and_display(self, step, target, pred, action, lr_rate, vetoes, spk_rec, vetoed):
        if not vetoed:
            self.recent_acc.append(1.0 if target == pred else 0.0)
        self.total_spikes += spk_rec.sum().item()
        acc = sum(self.recent_acc) / len(self.recent_acc) if self.recent_acc else 0
        
        # Display Actual Class Names instead of just numbers
        t_name = CIFAR_CLASSES[target]
        p_name = CIFAR_CLASSES[pred]
        
        print(f"Step {step:03d} | Target: {t_name:<5} | Pred: {p_name:<5} | "
              f"Action: {action:<12} | Acc: {acc*100:05.1f}% | LR: {lr_rate:.5f} | Vetoes: {vetoes}")

# ==========================================
# 6. INFERENCE TESTER
# ==========================================
def test_snn(net, device, num_steps=25, num_samples=5):
    print("\n" + "="*40)
    print("🧪 INITIATING CIFAR-10 INFERENCE PHASE")
    print("="*40)
    
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    net.eval()
    correct_count = 0
    
    with torch.no_grad(): 
        for i, (data, target) in enumerate(test_loader):
            if i >= num_samples: break
            data, target = data.to(device), target.to(device)
            target_val = target.item()
            
            print(f"\n--- Test Sample {i+1} ---")
            img_np = data.squeeze().cpu().numpy() #[3, 32, 32] RGB
            # Convert RGB to Grayscale for Terminal ASCII
            gray = 0.2989 * img_np[0] + 0.5870 * img_np[1] + 0.1140 * img_np[2]
            ascii_str = ""
            for row in gray:
                for pixel in row:
                    if pixel > 0.7: ascii_str += "██"
                    elif pixel > 0.4: ascii_str += "▓▓"
                    elif pixel > 0.2: ascii_str += "░░"
                    else: ascii_str += "  "
                ascii_str += "\n"
            print(ascii_str)

            # ConvNet doesn't need flattening, keep spatial dimensions
            spike_data = spikegen.rate(data, num_steps=num_steps)
            
            # UNPACK BOTH (Spikes and Voltage)
            spk_rec, mem_rec = net(spike_data) 
            
            # PREDICT USING VOLTAGE (Just like the training loop)
            total_voltage = mem_rec.sum(dim=0)
            prediction = total_voltage.argmax(dim=1).item()
            
            probs = torch.softmax(total_voltage.float(), dim=1)
            confidence = probs[0][prediction].item() * 100
            
            is_correct = (prediction == target_val)
            if is_correct: correct_count += 1
            
            status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
            print(f"Actual: {CIFAR_CLASSES[target_val]} | SNN Guessed: {CIFAR_CLASSES[prediction]} | Confidence: {confidence:.1f}% | {status}")
    print("\n" + "="*40)
    print(f"Test Accuracy on {num_samples} unseen samples: {(correct_count/num_samples)*100:.1f}%")

# ==========================================
# 7. MAIN LOOP
# ==========================================
def main():
    # CIFAR-10 Transform (No Normalization so spikegen.rate works safely between 0-1)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    net = SpikingVisualCortex().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0005, betas=(0.8, 0.999), weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    
    cog_map = CognitiveMap()
    curriculum_loader = CurriculumDataLoader(dataset)
    llm_cortex = LLMPrefrontalCortex()
    telemetry = TelemetryTracker()

    num_steps = 25 
    total_passes = 50
    steps_per_pass = 1000 # Increased because CIFAR is much harder than MNIST

    base_reward_lr = 0.0008
    base_unlearn_lr = 0.0003
    hippocampal_vault = []
    
    for pass_num in range(total_passes):
        print(f"\n{'='*40}\n☀️ WAKING UP: STARTING PASS {pass_num + 1}\n{'='*40}")
        plan_weights = llm_cortex.plan_curriculum(cog_map)
        
        for step, (data, target) in enumerate(curriculum_loader.yield_targeted_batch(plan_weights, steps_per_pass)):
            data, target = data.to(device), target.to(device)
            
            is_poisoned = False
            if random.random() < 0.05:
                data = data + torch.rand_like(data) * 0.9 
                target[0] = (target[0] + 5) % 10  
                is_poisoned = True

            # Data goes into ConvNet without view/flattening here
# --- 1. SENSORY PROCESSING ---
            spike_data = spikegen.rate(data, num_steps=num_steps)
            net.train()
            spk_rec, mem_rec = net(spike_data)
            
            total_voltage = mem_rec.sum(dim=0)
            prediction = total_voltage.argmax(dim=1).item()
            target_val = target.item()
            
            probs = torch.softmax(total_voltage.float(), dim=1)
            confidence = probs[0][prediction].item()

            action_taken = ""
            is_vetoed = False
            current_lr = optimizer.param_groups[0]['lr'] # Fallback for logging
            
            # --- 2. ANOMALY DETECTION (LLM) ---
            if is_poisoned and confidence > 0.7 and prediction != target_val:
                print(f"🚨 [HEURISTIC] Anomaly detected! Waking up LLM for evaluation...")
                is_vetoed = llm_cortex.evaluate_anomaly(prediction, target_val, confidence)

            # --- 3. NEUROMODULATION & LEARNING ---
            if is_vetoed:
                action_taken = "🛡️ VETOED"
                # NO BACKWARD PASS HERE. The LLM protected the brain from garbage data.
            else:
                base_lr = 0.0005 
                plasticity_factor = max(0.90 ** pass_num, 0.20) 
                
                if prediction == target_val:
                    # 🛑 METAPLASTICITY: Protect perfect synapses
                    if confidence > 0.85:
                        action_taken = "💤 RETAINED"
                        # NO BACKWARD PASS HERE. Save energy and protect perfection.
                    else:
                        # ✅ REWARD
                        dynamic_scale = (1.0 - confidence) 
                        current_lr = base_lr * plasticity_factor * (dynamic_scale + 0.1)
                        for param_group in optimizer.param_groups: param_group['lr'] = current_lr
                        
                        loss = loss_fn(total_voltage, target)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        action_taken = "✅ REWARDED"
                else:
                    # ❌ UNLEARN
                    dynamic_scale = confidence 
                    current_lr = base_lr * plasticity_factor * (dynamic_scale + 0.1)
                    for param_group in optimizer.param_groups: param_group['lr'] = current_lr
                    
                    loss = loss_fn(total_voltage, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    action_taken = "❌ UNLEARNED"
                    
                    # 🧠 SAVE TO VAULT: Core trauma (high confidence, but wrong)
                    if confidence > 0.6:
                        if len(hippocampal_vault) < 100: 
                            hippocampal_vault.append((data.detach(), target.detach()))
                        else:
                            hippocampal_vault[random.randint(0, 99)] = (data.detach(), target.detach())

                # Update the map ONLY if it wasn't vetoed by the LLM
                cog_map.update_map(target_val, correct=(prediction == target_val))

            # --- 4. TELEMETRY ---
            telemetry.log_and_display(step, target_val, prediction, action_taken, current_lr, llm_cortex.veto_count, spk_rec, is_vetoed)

        print(f"\n🌙 PASS {pass_num + 1} COMPLETE. Entering REM Sleep...")
        
        # Dream Sequence: Replay the hardest memories of the day 3 times (REM cycles)
        if len(hippocampal_vault) > 10:
            net.train()
            print(f"🧠 Dreaming about {len(hippocampal_vault)} difficult memories...")
            
            # Drop learning rate to an ultra-tiny level so dreams don't overwrite waking reality, 
            # they just gently carve the synapses deeper.
            for param_group in optimizer.param_groups: param_group['lr'] = 0.00005 
            
            for rem_cycle in range(3):
                random.shuffle(hippocampal_vault)
                
                # Review up to 30 hard memories per REM cycle
                for dream_data, dream_target in hippocampal_vault[:30]: 
                    spike_data = spikegen.rate(dream_data, num_steps=num_steps)
                    _, mem_rec = net(spike_data)
                    total_voltage = mem_rec.sum(dim=0)
                    
                    loss = loss_fn(total_voltage, dream_target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
        print("☀️ Waking up...")

    print("\n" + "="*40)
    print("🎓 CONTINUAL LEARNING SESSION FINISHED")
    print("Final Cognitive Map Mastery:")
    for k, v in cog_map.get_report().items(): print(f" - {k:<5}: {v}%")
    
    test_snn(net, device, num_steps=num_steps, num_samples=5)

if __name__ == "__main__":
    main()