import torch
import torch.nn as nn
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import spikegen
from collections import deque
import random
import requests
import json
import time
from snntorch import surrogate

# ==========================================
# 1. HARDWARE & CONFIG
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

print(f"🧠 Booting Neuro-Symbolic System on: {device}")
if device.type == 'cuda':
    print(f"💾 Initial VRAM Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")

# ==========================================
# 2. SNN ARCHITECTURE (The Reptilian Brain)
# ==========================================
class ContinuousSNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # SURROGATE GRADIENT: The "ghost gradient" that allows dead neurons to wake up
        # and learn even if they missed the spiking threshold.
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.fc1 = nn.Linear(28*28, 256)
        self.ln1 = nn.LayerNorm(256) # <--- CRITICAL: Prevents one neuron from dominating
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad) 
        
        self.fc2 = nn.Linear(256, 10)
        self.ln2 = nn.LayerNorm(10)  # <--- CRITICAL: Forces balanced output opportunities
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec =[]
        
        for step in range(x.size(0)):
            # Pass through Linear -> LayerNorm -> LIF Neuron
            cur1 = self.ln1(self.fc1(x[step])) 
            spk1, mem1 = self.lif1(cur1, mem1) 
            
            cur2 = self.ln2(self.fc2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spk2_rec.append(spk2)

        return torch.stack(spk2_rec)

# ==========================================
# 3. COGNITIVE MAP & CURRICULUM (Hippocampus)
# ==========================================
class CognitiveMap:
    def __init__(self, num_clusters=10):
        self.num_clusters = num_clusters
        self.cluster_mastery = torch.zeros(num_clusters) # 0.0 to 1.0

    def update_map(self, vector_id, correct):
        # Gentler penalty so the LLM doesn't panic and over-saturate a single digit
        learning_signal = 0.05 if correct else -0.05 
        self.cluster_mastery[vector_id] = torch.clamp(self.cluster_mastery[vector_id] + learning_signal, 0.0, 1.0)
        
    def get_report(self):
        # This is the function the LLM calls to read the map!
        return {f"Digit_{i}": round(self.cluster_mastery[i].item() * 100, 1) for i in range(self.num_clusters)}

class CurriculumDataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_by_cluster = {i:[] for i in range(10)}
        print("Organizing Embedded Vector Space...")
        for idx, (img, target) in enumerate(dataset):
            self.data_by_cluster[target].append(idx)

    def yield_targeted_batch(self, sampling_weights, steps=100):
        # Force a baseline of replay so mastered concepts are never totally ignored
        adjusted_weights = sampling_weights + 1.0 
        probs = adjusted_weights / adjusted_weights.sum()
        
        for _ in range(steps):
            chosen_cluster = torch.multinomial(probs, 1).item()
            idx = random.choice(self.data_by_cluster[chosen_cluster])
            yield self.dataset[idx][0].unsqueeze(0), torch.tensor([chosen_cluster])
            
# ==========================================
# 4. LLM PREFRONTAL CORTEX (Ollama Integration)
# ==========================================
class LLMPrefrontalCortex:
    def __init__(self):
        self.veto_count = 0

    def plan_curriculum(self, cognitive_map):
        print("\n🤔 [LLM] Analyzing Cognitive Map for Next Pass...")
        report = cognitive_map.get_report()
        
        prompt = f"""
        You are the prefrontal cortex of an AI. Here is the mastery percentage of the sensory network across 10 concepts (digits 0-9):
        {json.dumps(report)}
        
        Provide a sampling weight (1 to 10) for each concept to build a learning curriculum. 
        Give high weights to concepts with low mastery (needs practice). Give low weights (e.g. 1 or 2) to mastered concepts (to prevent forgetting).
        Reply ONLY with a valid JSON array of 10 integers. Example:[1, 5, 10, 2, 1, 8, 9, 1, 2, 3]
        """
        
        try:
            res = requests.post(OLLAMA_URL, json={"model": MODEL_NAME, "prompt": prompt, "stream": False, "format": "json"})
            weights = json.loads(res.json()['response'])
            if len(weights) == 10:
                print(f"📋 [LLM] New Curriculum Weights: {weights}\n")
                return torch.tensor(weights, dtype=torch.float32)
        except Exception as e:
            print(f"⚠️ LLM Curriculum failed. Defaulting to uniform weights. Error: {e}")
        
        return torch.ones(10)

    def evaluate_anomaly(self, SNN_pred, target, confidence):
        # Only called if the fast heuristic flags an anomaly (saving GPU time)
        prompt = f"""
        Sensory net predicted {SNN_pred} with {confidence*100:.1f}% confidence. The training label claims it is {target}.
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
# 5. TELEMETRY & METRICS
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
        print(f"Step {step:03d} | Target: {target} | Pred: {pred} | "
              f"Action: {action:<12} | Acc: {acc*100:05.1f}% | LR: {lr_rate:.5f} | Vetoes: {vetoes}")

def test_snn(net, device, num_steps=25, num_samples=5):
    print("\n" + "="*40)
    print("🧪 INITIATING INFERENCE / TESTING PHASE")
    print("="*40)
    
    # 1. Load the Unseen Test Dataset (train=False)
    transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # 2. Lock the Synapses (No Learning)
    net.eval()
    
    correct_count = 0
    
    with torch.no_grad(): # Disable weight updates
        for i, (data, target) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            data, target = data.to(device), target.to(device)
            target_val = target.item()
            
            # Print the image to the terminal using ASCII
            print(f"\n--- Test Sample {i+1} ---")
            img_np = data.squeeze().cpu().numpy()
            ascii_str = ""
            for row in img_np:
                for pixel in row:
                    if pixel > 0.7: ascii_str += "██"
                    elif pixel > 0.3: ascii_str += "░░"
                    else: ascii_str += "  "
                ascii_str += "\n"
            print(ascii_str)

            # SNN Processes the Image
            spike_data = spikegen.rate(data, num_steps=num_steps).view(num_steps, 1, -1)
            spk_rec = net(spike_data)
            
            # Count the spikes to get the network's answer
            total_spikes = spk_rec.sum(dim=0)
            prediction = total_spikes.argmax(dim=1).item()
            
            # Calculate Confidence
            probs = torch.softmax(total_spikes.float(), dim=1)
            confidence = probs[0][prediction].item() * 100
            
            # Result
            is_correct = (prediction == target_val)
            if is_correct: correct_count += 1
            
            status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
            print(f"Actual Digit: {target_val} | SNN Guessed: {prediction} | Confidence: {confidence:.1f}% | {status}")
            
    print("\n" + "="*40)
    print(f"Test Accuracy on {num_samples} unseen samples: {(correct_count/num_samples)*100:.1f}%")


# ==========================================
# 6. MAIN NEURO-SYMBOLIC LOOP
# ==========================================
def main():
    # Init Data & Models
    transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    net = ContinuousSNN().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    
    cog_map = CognitiveMap()
    curriculum_loader = CurriculumDataLoader(dataset)
    llm_cortex = LLMPrefrontalCortex()
    telemetry = TelemetryTracker()

    num_steps = 25 # 25ms biological simulation
    total_passes = 15
    steps_per_pass = 500 
    base_reward_lr = 0.0001
    base_unlearn_lr = 0.0005

    for pass_num in range(total_passes):
        print(f"\n{'='*40}\n☀️ WAKING UP: STARTING PASS {pass_num + 1}\n{'='*40}")
        
        # 1. LLM Plans the day based on memory consolidation
        plan_weights = llm_cortex.plan_curriculum(cog_map)
        
        # 2. Continuous Online Learning Loop
        for step, (data, target) in enumerate(curriculum_loader.yield_targeted_batch(plan_weights, steps_per_pass)):
            data, target = data.to(device), target.to(device)
            
            # Simulate Data Poisoning Anomaly (5% chance)
            is_poisoned = False
            if random.random() < 0.05:
                data = data + torch.rand_like(data) * 0.9 
                target[0] = (target[0] + 5) % 10  
                is_poisoned = True

            # SNN Senses the Environment
            spike_data = spikegen.rate(data, num_steps=num_steps).view(num_steps, 1, -1)
            net.train()
            spk_rec = net(spike_data)
            
            total_spikes = spk_rec.sum(dim=0)
            prediction = total_spikes.argmax(dim=1).item()
            target_val = target.item()
            
            # Calculate Confidence
            probs = torch.softmax(total_spikes.float(), dim=1)
            confidence = probs[0][prediction].item()

            # Fast Heuristic Anomaly Detection (Triggers LLM)
            action_taken = ""
            is_vetoed = False
            
            if is_poisoned and confidence > 0.7 and prediction != target_val:
                print(f"🚨 [HEURISTIC] Anomaly detected! Waking up LLM for evaluation...")
                is_vetoed = llm_cortex.evaluate_anomaly(prediction, target_val, confidence)

            if is_vetoed:
                action_taken = "🛡️ VETOED"
            else:
                # Calculate the brain's age (Plasticity drops by 15% every pass)
                plasticity_factor = 0.85 ** pass_num 
                
                if prediction == target_val:
                    # ✅ REWARD: Lock in memory 
                    current_lr = base_reward_lr * plasticity_factor
                    for param_group in optimizer.param_groups: param_group['lr'] = current_lr
                    loss = loss_fn(total_spikes.float(), target)
                    action_taken = "✅ REWARDED"
                else:
                    # ❌ UNLEARN: Correct the mistake 
                    current_lr = base_unlearn_lr * plasticity_factor
                    for param_group in optimizer.param_groups: param_group['lr'] = current_lr
                    loss = loss_fn(total_spikes.float(), target)
                    action_taken = "❌ UNLEARNED"
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update Vector Space Map
                cog_map.update_map(target_val, correct=(prediction == target_val))

            # Logging
            current_lr = optimizer.param_groups[0]['lr']
            telemetry.log_and_display(step, target_val, prediction, action_taken, current_lr, llm_cortex.veto_count, spk_rec, is_vetoed)


        print(f"\n🌙 PASS {pass_num + 1} COMPLETE. Network is entering sleep cycle to consolidate memories.")
        time.sleep(2) # Simulate sleep

    print("\n" + "="*40)
    print("🎓 CONTINUAL LEARNING SESSION FINISHED")
    print("Final Cognitive Map Mastery:")
    for k, v in cog_map.get_report().items(): print(f" - {k}: {v}%")
    print(f"Total Energy Cost (Spikes Fired): {telemetry.total_spikes}")
    print("="*40)
    test_snn(net, device, num_steps=num_steps, num_samples=5)

if __name__ == "__main__":
    main()