"""
GIKI Chatbot MCP Server
Implements Model Context Protocol with external API tools
"""

import os
import sys
import json
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pathlib import Path
import requests

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env from parent directory
env_path = Path(__file__).resolve().parent.parent / '.env'
logger.info(f"MCP Server looking for .env at: {env_path}")
logger.info(f"MCP Server .env exists: {env_path.exists()}")

if not env_path.exists():
    print("❌ .env file not found at:", env_path)
    print("Create .env file with your API keys first")
    exit(1)

loaded = load_dotenv(dotenv_path=env_path)
logger.info(f"MCP Server loaded .env: {loaded}")
logger.info(f"MCP Server API keys present: {', '.join(k for k in os.environ if '_API_KEY' in k)}")
logger = logging.getLogger(__name__)


class MCPTool:
    """Base class for MCP tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool"""
        raise NotImplementedError


class WeatherTool(MCPTool):
    """Get current weather at GIKI campus"""
    
    def __init__(self):
        super().__init__(
            name="get_giki_weather", 
            description="Get current weather conditions at GIKI campus"
        )
        # Add debug logging
        logger.info("Initializing WeatherTool")
        
        # Add stack trace logging
        import traceback
        logger.info("WeatherTool initialization stack trace:")
        logger.info(''.join(traceback.format_stack()))
        
        logger.info(f"Environment variables: {', '.join(k for k in os.environ if '_API_KEY' in k or 'GIKI_' in k)}")
        
        self.api_key = os.getenv('WEATHER_API_KEY')
        logger.info(f"Weather API key present: {bool(self.api_key)}")
        
        if not self.api_key:
            logger.error("WEATHER_API_KEY not found in environment variables")
            raise ValueError("WEATHER_API_KEY not found in environment variables")
            
        self.lat = float(os.getenv('GIKI_LATITUDE', 33.9407))
        self.lon = float(os.getenv('GIKI_LONGITUDE', 72.6267))
    
    async def execute(self) -> Dict[str, Any]:
        """Get weather data"""
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': self.lat,
                'lon': self.lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            return {
                'success': True,
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'location': 'GIKI Campus, Swabi',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return {
                'success': False,
                'error': str(e)
            }


class PrayerTimesTool(MCPTool):
    """Get Islamic prayer times for GIKI location"""
    
    def __init__(self):
        super().__init__(
            name="get_prayer_times",
            description="Get today's Islamic prayer times for GIKI campus"
        )
        self.city = os.getenv('GIKI_CITY', 'Swabi')
        self.country = os.getenv('GIKI_COUNTRY', 'Pakistan')
    
    async def execute(self) -> Dict[str, Any]:
        """Get prayer times"""
        try:
            url = "http://api.aladhan.com/v1/timingsByCity"
            params = {
                'city': self.city,
                'country': self.country,
                'method': 1  # University of Islamic Sciences, Karachi
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            timings = data['data']['timings']
            
            return {
                'success': True,
                'fajr': timings['Fajr'],
                'dhuhr': timings['Dhuhr'],
                'asr': timings['Asr'],
                'maghrib': timings['Maghrib'],
                'isha': timings['Isha'],
                'date': data['data']['date']['readable'],
                'location': f"{self.city}, {self.country}"
            }
        except Exception as e:
            logger.error(f"Prayer times API error: {e}")
            return {
                'success': False,
                'error': str(e)
            }


class CurrencyConverterTool(MCPTool):
    """Convert currency for international students"""
    
    def __init__(self):
        super().__init__(
            name="convert_currency",
            description="Convert between currencies (useful for international students calculating fees)"
        )
        self.api_key = os.getenv('EXCHANGE_RATE_API_KEY')
        if not self.api_key:
            logger.error("EXCHANGE_RATE_API_KEY not found in environment variables")
            raise ValueError("EXCHANGE_RATE_API_KEY not found in environment variables")
    
    async def execute(self, amount: float, from_currency: str = "USD", to_currency: str = "PKR") -> Dict[str, Any]:
        """Convert currency"""
        try:
            url = f"https://v6.exchangerate-api.com/v6/{self.api_key}/pair/{from_currency}/{to_currency}/{amount}"
            
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if data['result'] == 'success':
                return {
                    'success': True,
                    'from_amount': amount,
                    'from_currency': from_currency,
                    'to_amount': data['conversion_result'],
                    'to_currency': to_currency,
                    'exchange_rate': data['conversion_rate'],
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': 'Currency conversion failed'
                }
        except Exception as e:
            logger.error(f"Currency API error: {e}")
            return {
                'success': False,
                'error': str(e)
            }


class NewsTool(MCPTool):
    """Get latest Pakistan tech/education news"""
    
    def __init__(self):
        super().__init__(
            name="get_pakistan_tech_news",
            description="Get latest technology and education news from Pakistan"
        )
        self.api_key = os.getenv('NEWS_API_KEY')
        if not self.api_key:
            logger.error("NEWS_API_KEY not found in environment variables")
            raise ValueError("NEWS_API_KEY not found in environment variables")
    
    async def execute(self, category: str = "technology", limit: int = 5) -> Dict[str, Any]:
        """Get news"""
        try:
            url = "https://newsapi.org/v2/top-headlines"
            params = {
                'country': 'pk',
                'category': category,
                'apiKey': self.api_key,
                'pageSize': limit
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'ok':
                articles = []
                for article in data['articles'][:limit]:
                    articles.append({
                        'title': article['title'],
                        'description': article.get('description', ''),
                        'url': article['url'],
                        'source': article['source']['name'],
                        'published_at': article['publishedAt']
                    })
                
                return {
                    'success': True,
                    'articles': articles,
                    'total': len(articles),
                    'category': category
                }
            else:
                return {
                    'success': False,
                    'error': 'News fetch failed'
                }
        except Exception as e:
            logger.error(f"News API error: {e}")
            return {
                'success': False,
                'error': str(e)
            }


class FeeCalculatorTool(MCPTool):
    """Calculate GIKI fees"""
    
    def __init__(self):
        super().__init__(
            name="calculate_giki_fees",
            description="Calculate total GIKI fees based on semesters, scholarship, and hostel"
        )
        # GIKI fee structure (per semester)
        self.tuition_per_semester = 250000  # PKR
        self.hostel_per_semester = 50000    # PKR
        self.security_deposit = 25000       # One-time
    
    async def execute(self, 
                     semesters: int = 8, 
                     scholarship_percentage: int = 0,
                     include_hostel: bool = True) -> Dict[str, Any]:
        """Calculate fees"""
        try:
            # Calculate tuition
            total_tuition = self.tuition_per_semester * semesters
            
            # Apply scholarship
            scholarship_amount = (total_tuition * scholarship_percentage) / 100
            net_tuition = total_tuition - scholarship_amount
            
            # Add hostel if needed
            hostel_cost = 0
            if include_hostel:
                hostel_cost = self.hostel_per_semester * semesters
            
            # Total
            total_cost = net_tuition + hostel_cost + self.security_deposit
            
            return {
                'success': True,
                'breakdown': {
                    'gross_tuition': total_tuition,
                    'scholarship_percentage': scholarship_percentage,
                    'scholarship_amount': scholarship_amount,
                    'net_tuition': net_tuition,
                    'hostel_charges': hostel_cost,
                    'security_deposit': self.security_deposit,
                    'total_cost': total_cost
                },
                'semesters': semesters,
                'years': semesters / 2,
                'per_semester_average': total_cost / semesters
            }
        except Exception as e:
            logger.error(f"Fee calculation error: {e}")
            return {
                'success': False,
                'error': str(e)
            }


class EventsTool(MCPTool):
    """Get upcoming GIKI events"""
    
    def __init__(self):
        super().__init__(
            name="get_giki_events",
            description="Get upcoming events and activities at GIKI"
        )
    
    async def execute(self) -> Dict[str, Any]:
        """Get events from internal database"""
        try:
            # This would normally come from a database
            # For now, returning mock data
            events = [
                {
                    'name': 'GIKI Tech Fest 2025',
                    'date': '2025-03-15',
                    'location': 'Main Auditorium',
                    'description': 'Annual technology festival with competitions and exhibitions',
                    'category': 'Technical'
                },
                {
                    'name': 'Career Fair',
                    'date': '2025-03-20',
                    'location': 'Sports Complex',
                    'description': 'Meet recruiters from top tech companies',
                    'category': 'Career'
                },
                {
                    'name': 'Hackathon 2025',
                    'date': '2025-03-25',
                    'location': 'Computer Labs',
                    'description': '24-hour coding competition',
                    'category': 'Technical'
                },
                {
                    'name': 'Sports Gala',
                    'date': '2025-04-05',
                    'location': 'Sports Grounds',
                    'description': 'Inter-department sports competition',
                    'category': 'Sports'
                },
                {
                    'name': 'Cultural Night',
                    'date': '2025-04-15',
                    'location': 'Open Air Theatre',
                    'description': 'Music, drama, and cultural performances',
                    'category': 'Cultural'
                }
            ]
            
            return {
                'success': True,
                'events': events,
                'total': len(events)
            }
        except Exception as e:
            logger.error(f"Events error: {e}")
            return {
                'success': False,
                'error': str(e)
            }


class MCPServer:
    """Main MCP Server managing all tools"""
    
    def __init__(self):
        # Verify environment variables are available
        required_vars = ['WEATHER_API_KEY', 'EXCHANGE_RATE_API_KEY', 'NEWS_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Initialize tools
        self.tools = {
            'get_giki_weather': WeatherTool(),
            'get_prayer_times': PrayerTimesTool(),
            'convert_currency': CurrencyConverterTool(),
            'get_pakistan_tech_news': NewsTool(),
            'calculate_giki_fees': FeeCalculatorTool(),
            'get_giki_events': EventsTool()
        }
        logger.info(f"MCP Server initialized with {len(self.tools)} tools")
    
    def list_tools(self) -> List[Dict[str, str]]:
        """List all available tools"""
        return [
            {
                'name': tool.name,
                'description': tool.description
            }
            for tool in self.tools.values()
        ]
    
    async def call_tool(self, tool_name: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a tool"""
        if tool_name not in self.tools:
            return {
                'success': False,
                'error': f"Tool '{tool_name}' not found"
            }
        
        try:
            tool = self.tools[tool_name]
            if params:
                result = await tool.execute(**params)
            else:
                result = await tool.execute()
            return result
        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict]:
        """Get tool parameter schema"""
        schemas = {
            'convert_currency': {
                'parameters': {
                    'amount': {'type': 'number', 'required': True, 'description': 'Amount to convert'},
                    'from_currency': {'type': 'string', 'required': False, 'default': 'USD'},
                    'to_currency': {'type': 'string', 'required': False, 'default': 'PKR'}
                }
            },
            'get_pakistan_tech_news': {
                'parameters': {
                    'category': {'type': 'string', 'required': False, 'default': 'technology'},
                    'limit': {'type': 'integer', 'required': False, 'default': 5}
                }
            },
            'calculate_giki_fees': {
                'parameters': {
                    'semesters': {'type': 'integer', 'required': False, 'default': 8},
                    'scholarship_percentage': {'type': 'integer', 'required': False, 'default': 0},
                    'include_hostel': {'type': 'boolean', 'required': False, 'default': True}
                }
            }
        }
        return schemas.get(tool_name)


# Global MCP server instance
mcp_server = MCPServer()


# Initialize singleton instance
if not hasattr(sys.modules[__name__], 'mcp_server'):
    logger.info("Creating new MCP Server instance")
    mcp_server = MCPServer()
else:
    logger.info("Using existing MCP Server instance")

# Test function
async def test_mcp_server():
    """Test all MCP tools"""
    print("\n" + "="*70)
    print("🧪 TESTING MCP SERVER")
    print("="*70 + "\n")
    
    # Test weather
    print("1. Testing Weather Tool...")
    result = await mcp_server.call_tool('get_giki_weather')
    print(f"   Result: {json.dumps(result, indent=2)}\n")
    
    # Test prayer times
    print("2. Testing Prayer Times Tool...")
    result = await mcp_server.call_tool('get_prayer_times')
    print(f"   Result: {json.dumps(result, indent=2)}\n")
    
    # Test currency
    print("3. Testing Currency Converter...")
    result = await mcp_server.call_tool('convert_currency', {'amount': 1000, 'from_currency': 'USD', 'to_currency': 'PKR'})
    print(f"   Result: {json.dumps(result, indent=2)}\n")
    
    # Test news
    print("4. Testing News Tool...")
    result = await mcp_server.call_tool('get_pakistan_tech_news', {'limit': 3})
    print(f"   Result: {json.dumps(result, indent=2)}\n")
    
    # Test fee calculator
    print("5. Testing Fee Calculator...")
    result = await mcp_server.call_tool('calculate_giki_fees', {'semesters': 8, 'scholarship_percentage': 25, 'include_hostel': True})
    print(f"   Result: {json.dumps(result, indent=2)}\n")
    
    # Test events
    print("6. Testing Events Tool...")
    result = await mcp_server.call_tool('get_giki_events')
    print(f"   Result: {json.dumps(result, indent=2)}\n")
    
    print("="*70)
    print("✅ ALL TESTS COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_mcp_server())