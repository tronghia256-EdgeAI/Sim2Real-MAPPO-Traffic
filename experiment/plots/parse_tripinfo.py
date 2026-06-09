import xml.etree.ElementTree as ET
import os

def get_throughput_and_co2(tripinfo_path):
    """Parse SUMO tripinfo.xml to extract throughput and CO2 emissions."""
    if not os.path.exists(tripinfo_path):
        print(f"❌ Error: File not found at:\n{tripinfo_path}")
        return 0, 0

    try:
        tree = ET.parse(tripinfo_path)
        root = tree.getroot()
    except Exception as e:
        print(f"❌ Error parsing XML: {e}")
        return 0, 0

    throughput = 0
    total_co2_mg = 0.0

    # Iterate over completed trips
    for trip in root.findall('tripinfo'):
        throughput += 1
        
        # Extract CO2 emissions
        emissions = trip.find('emissions')
        if emissions is not None:
            co2_val = emissions.get('CO2_abs')
            if co2_val:
                total_co2_mg += float(co2_val)

    # Convert mg to kg for reporting
    total_co2_kg = total_co2_mg / 1_000_000
    
    return throughput, total_co2_kg


if __name__ == "__main__":
    print("⏳ Extracting data from SUMO logs...\n")

    from pathlib import Path
    _root = Path(__file__).resolve().parents[2]
    eval_dir = _root / "sumo_configs" / "evaluation" / "medium"

    baseline_file = str(eval_dir / "tripinfo_baseline.xml")
    ppo_file = str(eval_dir / "tripinfo_ppo.xml")

    # 1. Evaluate Baseline
    print("="*45)
    print("🔵 BASELINE (FIXED-TIME) METRICS")
    print("="*45)
    thpt_base, co2_base = get_throughput_and_co2(baseline_file)
    if thpt_base > 0:
        print(f"🚦 Throughput: {thpt_base} vehicles")
        print(f"☁️ CO2 Emissions: {co2_base:.2f} kg\n")

    # 2. Evaluate PPO 
    print("="*45)
    print("🟢 PPO (AI) METRICS")
    print("="*45)
    thpt_ppo, co2_ppo = get_throughput_and_co2(ppo_file)
    if thpt_ppo > 0:
        print(f"🚦 Throughput: {thpt_ppo} vehicles")
        print(f"☁️ CO2 Emissions: {co2_ppo:.2f} kg\n")