"""
Setup Script for RAG System
Automated setup and installation
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def main():
    print("🚀 RAG System Setup")
    print("="*50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ is required")
        return
    
    print(f"✅ Python {python_version.major}.{python_version.minor} detected")
    
    # Create virtual environment (optional)
    create_venv = input("\n🔧 Create virtual environment? (y/n): ").lower().strip() == 'y'
    
    if create_venv:
        venv_path = Path("./venv")
        if not venv_path.exists():
            if not run_command("python -m venv venv", "Creating virtual environment"):
                return
        
        # Activation instructions
        if os.name == 'nt':  # Windows
            activate_cmd = ".\\venv\\Scripts\\activate"
        else:  # Unix/MacOS
            activate_cmd = "source venv/bin/activate"
        
        print(f"💡 To activate virtual environment, run: {activate_cmd}")
        
        # Ask if user wants to continue without activation
        continue_setup = input("⚠️  Continue setup without activating venv? (y/n): ").lower().strip() == 'y'
        if not continue_setup:
            print("🔧 Please activate the virtual environment and run this script again")
            return
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        if not run_command("pip install -r requirements.txt", "Installing requirements"):
            print("⚠️  Some packages may have failed to install")
            print("💡 You may need to install them individually")
    else:
        print("❌ requirements.txt not found")
        return
    
    # Download NLTK data
    print("\n📚 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded")
    except ImportError:
        print("⚠️  NLTK not installed, skipping data download")
    except Exception as e:
        print(f"⚠️  NLTK data download failed: {e}")
    
    # Create directories
    print("\n📁 Creating directories...")
    
    directories = [
        "data/documents",
        "data/vectordb", 
        "data/cache",
        "data/processed",
        "logs",
        "temp_uploads"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create .env file
    print("\n🔐 Setting up environment...")
    
    env_file = Path(".env")
    if not env_file.exists():
        # Copy from example
        env_example = Path(".env.example")
        if env_example.exists():
            with open(env_example, 'r') as f:
                env_content = f.read()
            
            with open(env_file, 'w') as f:
                f.write(env_content)
            
            print("✅ Created .env file from template")
            print("🔧 Please edit .env file and add your Gemini API key")
        else:
            print("⚠️  .env.example not found")
    else:
        print("✅ .env file already exists")
    
    # Test imports
    print("\n🧪 Testing imports...")
    
    test_imports = [
        ("google.generativeai", "Google Gemini API"),
        ("chromadb", "ChromaDB"),
        ("streamlit", "Streamlit"),
        ("PyPDF2", "PDF processing"),
        ("docx", "DOCX processing"),
        ("markdown", "Markdown processing"),
        ("nltk", "Natural Language Toolkit"),
        ("sentence_transformers", "Sentence Transformers"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas")
    ]
    
    failed_imports = []
    
    for module, description in test_imports:
        try:
            __import__(module)
            print(f"✅ {description}")
        except ImportError:
            print(f"❌ {description} - Not installed")
            failed_imports.append(module)
    
    # API key check
    print("\n🔑 API Key Configuration:")
    
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        print("✅ GEMINI_API_KEY found in environment")
    else:
        print("⚠️  GEMINI_API_KEY not found in environment")
        print("🔧 You can:")
        print("   1. Set it in the .env file")
        print("   2. Set it as an environment variable")
        print("   3. Enter it when prompted by the application")
    
    # Setup summary
    print("\n" + "="*50)
    print("📋 Setup Summary")
    print("="*50)
    
    if failed_imports:
        print(f"⚠️  {len(failed_imports)} packages failed to import:")
        for module in failed_imports:
            print(f"   - {module}")
        print("\n💡 Try installing missing packages individually:")
        for module in failed_imports:
            print(f"   pip install {module}")
    else:
        print("✅ All required packages imported successfully")
    
    print("\n🚀 Next Steps:")
    print("1. Edit .env file and add your Gemini API key")
    print("2. Run the web interface: streamlit run app.py")
    print("3. Or try the CLI: python cli.py --help")
    print("4. Or run quick start: python examples/quick_start.py")
    
    print("\n📚 Documentation:")
    print("- Basic usage: python examples/quick_start.py")
    print("- Advanced features: python examples/advanced_features.py")
    print("- Web interface: streamlit run app.py")
    print("- Command line: python cli.py --help")
    
    if api_key:
        print(f"\n✅ Setup completed successfully!")
    else:
        print(f"\n⚠️  Setup completed - Don't forget to configure your API key!")

if __name__ == "__main__":
    main()