import os

folders = ["data", "notebooks", "scripts", "models", "dashboard"]

base_path = "."

for folder in folders:
    os.makedirs(os.path.join(base_path, folder), exist_ok=True)

with open(os.path.join(base_path, "README.md"), "w") as f:
    f.write("# AI-Driven Coastal Security\n\nProject initialized.\n")

with open(os.path.join(base_path, "requirements.txt"), "w") as f:
    f.write("""pandas==2.2.1
numpy==1.26.4
scikit-learn==1.4.2
matplotlib==3.8.4
seaborn==0.13.2
jupyter==1.0.0
notebook==7.1.3
""")

print("✅ Folder structure and base files created!")
