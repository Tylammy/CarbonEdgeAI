import random

def get_renewable_energy_availability():
    # Simulate renewable energy availability (0-100%)
    return random.randint(50, 100)

def offload_to_cloud(model_task):
    availability = get_renewable_energy_availability()
    if availability > 70:
        print(f"Offloading {model_task} to renewable-energy-powered cloud data center...")
        # Simulate offloading
        print(f"{model_task} successfully completed using {availability}% renewable energy.")
    else:
        print(f"Energy availability too low ({availability}%). Processing {model_task} on Edge.")
        # Process on Edge instead
        print(f"{model_task} processed locally on Edge.")
        
if __name__ == "__main__":
    model_task = "image classification"
    offload_to_cloud(model_task)
