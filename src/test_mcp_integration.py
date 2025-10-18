"""
Test MCP Integration
Run this to verify all MCP tools work with your API keys
"""

import asyncio
import sys
import os
from dotenv import load_dotenv
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env from parent directory
env_path = Path(__file__).resolve().parent.parent / '.env'
logger.info(f"Looking for .env file at: {env_path}")
logger.info(f"File exists: {env_path.exists()}")

if not env_path.exists():
    print("❌ .env file not found at:", env_path)
    print("Create .env file with your API keys first")
    sys.exit(1)

loaded = load_dotenv(dotenv_path=env_path)
logger.info(f"Loaded .env file successfully: {loaded}")
logger.info(f"WEATHER_API_KEY present: {'WEATHER_API_KEY' in os.environ}")

# Import MCP server
try:
    print("🔄 Importing MCP server...")
    from mcp_server import mcp_server
    print("✅ MCP server imported successfully")
except ImportError as e:
    print("❌ Error: mcp_server.py not found!")
    print("Make sure mcp_server.py is in the same directory")
    print(f"Import error details: {e}")
    sys.exit(1)
except Exception as e:
    print("❌ Unexpected error during import!")
    import traceback
    print("Error details:", str(e))
    print("\nFull stack trace:")
    traceback.print_exc()
    sys.exit(1)

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'


async def test_all_tools():
    """Test all MCP tools"""
    
    print(f"\n{BLUE}{'='*70}")
    print("🧪 TESTING MCP TOOLS WITH YOUR API KEYS")
    print(f"{'='*70}{RESET}\n")
    
    results = {}
    
    # Test 1: Weather
    print(f"{CYAN}[1/6] Testing Weather Tool...{RESET}")
    try:
        result = await mcp_server.call_tool('get_giki_weather')
        if result.get('success'):
            print(f"{GREEN}✓ Weather API working!{RESET}")
            print(f"   Temperature: {result['temperature']}°C")
            print(f"   Conditions: {result['description']}")
            results['weather'] = True
        else:
            print(f"{RED}✗ Weather API failed: {result.get('error')}{RESET}")
            results['weather'] = False
    except Exception as e:
        print(f"{RED}✗ Error: {e}{RESET}")
        results['weather'] = False
    
    # Test 2: Prayer Times
    print(f"\n{CYAN}[2/6] Testing Prayer Times Tool...{RESET}")
    try:
        result = await mcp_server.call_tool('get_prayer_times')
        if result.get('success'):
            print(f"{GREEN}✓ Prayer Times API working!{RESET}")
            print(f"   Fajr: {result['fajr']}, Zuhr: {result['dhuhr']}")
            results['prayer'] = True
        else:
            print(f"{RED}✗ Prayer Times failed: {result.get('error')}{RESET}")
            results['prayer'] = False
    except Exception as e:
        print(f"{RED}✗ Error: {e}{RESET}")
        results['prayer'] = False
    
    # Test 3: Currency Converter
    print(f"\n{CYAN}[3/6] Testing Currency Converter...{RESET}")
    try:
        result = await mcp_server.call_tool('convert_currency', {
            'amount': 1000,
            'from_currency': 'USD',
            'to_currency': 'PKR'
        })
        if result.get('success'):
            print(f"{GREEN}✓ Currency API working!{RESET}")
            print(f"   $1000 USD = PKR {result['to_amount']:,.2f}")
            results['currency'] = True
        else:
            print(f"{RED}✗ Currency API failed: {result.get('error')}{RESET}")
            results['currency'] = False
    except Exception as e:
        print(f"{RED}✗ Error: {e}{RESET}")
        results['currency'] = False
    
    # Test 4: News
    print(f"\n{CYAN}[4/6] Testing News Tool...{RESET}")
    try:
        result = await mcp_server.call_tool('get_pakistan_tech_news', {'limit': 3})
        if result.get('success'):
            print(f"{GREEN}✓ News API working!{RESET}")
            print(f"   Found {result['total']} articles")
            if result.get('articles'):
                print(f"   Top: {result['articles'][0]['title'][:60]}...")
            results['news'] = True
        else:
            print(f"{RED}✗ News API failed: {result.get('error')}{RESET}")
            results['news'] = False
    except Exception as e:
        print(f"{RED}✗ Error: {e}{RESET}")
        results['news'] = False
    
    # Test 5: Fee Calculator
    print(f"\n{CYAN}[5/6] Testing Fee Calculator...{RESET}")
    try:
        result = await mcp_server.call_tool('calculate_giki_fees', {
            'semesters': 8,
            'scholarship_percentage': 25,
            'include_hostel': True
        })
        if result.get('success'):
            print(f"{GREEN}✓ Fee Calculator working!{RESET}")
            print(f"   Total: PKR {result['breakdown']['total_cost']:,.0f}")
            results['fees'] = True
        else:
            print(f"{RED}✗ Fee Calculator failed: {result.get('error')}{RESET}")
            results['fees'] = False
    except Exception as e:
        print(f"{RED}✗ Error: {e}{RESET}")
        results['fees'] = False
    
    # Test 6: Events
    print(f"\n{CYAN}[6/6] Testing Events Tool...{RESET}")
    try:
        result = await mcp_server.call_tool('get_giki_events')
        if result.get('success'):
            print(f"{GREEN}✓ Events tool working!{RESET}")
            print(f"   Upcoming events: {result['total']}")
            results['events'] = True
        else:
            print(f"{RED}✗ Events failed: {result.get('error')}{RESET}")
            results['events'] = False
    except Exception as e:
        print(f"{RED}✗ Error: {e}{RESET}")
        results['events'] = False
    
    # Summary
    print(f"\n{BLUE}{'='*70}")
    print("📊 SUMMARY")
    print(f"{'='*70}{RESET}\n")
    
    working = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"✓ Working: {GREEN}{working}/{total}{RESET}")
    print(f"✗ Failed: {RED}{total - working}/{total}{RESET}\n")
    
    if working >= 4:
        print(f"{GREEN}🎉 Excellent! Most tools are working!{RESET}")
    elif working >= 2:
        print(f"{YELLOW}⚠ Some tools working, check failed ones{RESET}")
    else:
        print(f"{RED}❌ Most tools failed, check API keys in .env{RESET}")
    
    print(f"\n{BLUE}{'='*70}")
    print("💡 NEXT STEPS")
    print(f"{'='*70}{RESET}\n")
    
    if working >= 3:
        print("1. ✅ MCP integration is ready!")
        print("2. 🚀 Start the backend: python backend.py")
        print("3. 🌐 Start the frontend: python -m http.server 8000")
        print("4. 🧪 Test queries:")
        print("   • 'What's the weather at GIKI?'")
        print("   • 'When is Zuhr prayer?'")
        print("   • 'Convert $500 to PKR'")
        print("   • 'Latest tech news'")
        print("   • 'Calculate fee for 4 years'")
    else:
        print("1. ❌ Some tools need attention")
        print("2. 📝 Check .env file has correct API keys:")
        print("   WEATHER_API_KEY=84870b1aa0f9adcf87968eaf8892e07d")
        print("   EXCHANGE_RATE_API_KEY=6b9b99243934a69659fcf5ee")
        print("   NEWS_API_KEY=f29a7844397341b689bd0b5153e788bd")
        print("3. 🔄 Run this test again after fixing")
    
    print(f"\n{BLUE}{'='*70}{RESET}\n")


def main():
    """Main function"""
    # Environment variables should already be loaded from parent directory
    if not any(key in os.environ for key in ['WEATHER_API_KEY', 'EXCHANGE_RATE_API_KEY', 'NEWS_API_KEY']):
        print(f"{RED}❌ Environment variables not found!{RESET}")
        print("Make sure .env file exists with correct API keys")
        return
    
    # Check API keys
    weather_key = os.getenv('WEATHER_API_KEY')
    currency_key = os.getenv('EXCHANGE_RATE_API_KEY')
    news_key = os.getenv('NEWS_API_KEY')
    
    if not weather_key or not currency_key or not news_key:
        print(f"{YELLOW}⚠ Warning: Some API keys missing in .env{RESET}")
        print("\nExpected keys:")
        print(f"  WEATHER_API_KEY: {'✓' if weather_key else '✗'}")
        print(f"  EXCHANGE_RATE_API_KEY: {'✓' if currency_key else '✗'}")
        print(f"  NEWS_API_KEY: {'✓' if news_key else '✗'}")
        print("\nContinuing with available keys...\n")
    
    # Run tests
    asyncio.run(test_all_tools())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test cancelled")
    except Exception as e:
        print(f"\n{RED}❌ Error: {e}{RESET}")
        import traceback
        traceback.print_exc()