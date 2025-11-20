"""
Environment setup script for the fact verification system.
Run this to check configuration and set up your environment.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import validate_config, get_config, OPENAI_API_KEY


def setup_env_file():
    """Create .env file from example if it doesn't exist."""
    env_file = project_root / ".env"
    env_example = project_root / ".env.example"
    
    if not env_file.exists() and env_example.exists():
        print("📝 Creating .env file from example...")
        
        # Copy example to .env
        with open(env_example, 'r') as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Created {env_file}")
        print("📝 Please edit .env file and add your OpenAI API key")
        return True
    
    elif env_file.exists():
        print(f"✅ .env file exists at {env_file}")
        return False
    
    else:
        print("❌ No .env.example file found")
        return False


def check_openai_key():
    """Check if OpenAI API key is available."""
    if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here":
        print("✅ OpenAI API key is set")
        # Mask the key for display
        masked_key = OPENAI_API_KEY[:10] + "..." + OPENAI_API_KEY[-4:] if len(OPENAI_API_KEY) > 14 else "***"
        print(f"   Key: {masked_key}")
        return True
    else:
        print("❌ OpenAI API key not set or using placeholder value")
        print("   Set your API key in the .env file or environment variable")
        return False


def set_api_key_interactively():
    """Allow user to set API key interactively."""
    api_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
    
    if api_key:
        # Set for current session
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Update .env file
        env_file = project_root / ".env"
        if env_file.exists():
            with open(env_file, 'r') as f:
                content = f.read()
            
            # Replace the placeholder
            content = content.replace(
                "OPENAI_API_KEY=your_openai_api_key_here",
                f"OPENAI_API_KEY={api_key}"
            )
            
            with open(env_file, 'w') as f:
                f.write(content)
            
            print("✅ API key saved to .env file")
            return True
    
    return False


def main():
    """Main setup function."""
    print("🔧 FACT VERIFICATION SYSTEM - ENVIRONMENT SETUP")
    print("=" * 60)
    
    # Step 1: Set up .env file
    env_created = setup_env_file()
    
    # Step 2: Check configuration
    print("\n📋 Configuration Check:")
    print("-" * 30)
    
    config = get_config()
    for key, value in config.items():
        if key == "openai_api_key_set":
            status = "✅" if value else "❌"
            print(f"{status} OpenAI API Key: {'Set' if value else 'Not set'}")
        else:
            print(f"   {key}: {value}")
    
    # Step 3: Validate configuration
    print("\n🔍 Validation Results:")
    print("-" * 30)
    
    issues = validate_config()
    if not issues:
        print("✅ All configuration checks passed!")
    else:
        for key, issue in issues.items():
            print(f"❌ {issue}")
    
    # Step 4: Check OpenAI key specifically
    print("\n🔑 OpenAI API Key Status:")
    print("-" * 30)
    
    key_ok = check_openai_key()
    
    # Step 5: Offer to set key interactively
    if not key_ok and env_created:
        print("\n💡 You can set your OpenAI API key now:")
        set_api_key_interactively()
    
    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETE")
    print("=" * 60)
    
    if not issues or (len(issues) == 1 and "openai_api_key" in issues):
        print("\n🚀 Ready to run the fact verification system!")
        print("Next steps:")
        print("  1. python test_system.py  (test the system)")
        print("  2. streamlit run app.py   (run the web interface)")
    else:
        print("\n⚠️  Please fix the configuration issues above before proceeding.")


if __name__ == "__main__":
    main()