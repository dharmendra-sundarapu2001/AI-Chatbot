import os
import sys
import requests
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class TripPlanningAgent:
    """Main class for vacation trip planning functionality"""
    
    def __init__(self):
        """Initialize the trip planning agent with API keys and spell checker"""
        self.geoapify_api_key = os.getenv('GEOAPIFY_API_KEY')
        self.ors_api_key = os.getenv('ORS_API_KEY')
        
        if not self.geoapify_api_key:
            logger.error("GEOAPIFY_API_KEY environment variable not found")
            sys.exit(1)
        
        if not self.ors_api_key:
            logger.error("ORS_API_KEY environment variable not found")
            sys.exit(1)
        
        # Initialize spell checker and location database
        self._initialize_location_correction_system()
        
        logger.info("🚀 Trip Planning Agent initialized successfully with intelligent auto-correction")

    def _initialize_location_correction_system(self):
        """Initialize the intelligent location correction system"""
        # Create a comprehensive database of known Indian locations for fuzzy matching
        self.known_locations = {
            # Major cities
            'mumbai': 'Mumbai, Maharashtra',
            'delhi': 'New Delhi, Delhi',
            'bangalore': 'Bengaluru, Karnataka',
            'bengaluru': 'Bengaluru, Karnataka',
            'kolkata': 'Kolkata, West Bengal',
            'chennai': 'Chennai, Tamil Nadu',
            'hyderabad': 'Hyderabad, Telangana',
            'pune': 'Pune, Maharashtra',
            'ahmedabad': 'Ahmedabad, Gujarat',
            'jaipur': 'Jaipur, Rajasthan',
            'lucknow': 'Lucknow, Uttar Pradesh',
            'kanpur': 'Kanpur, Uttar Pradesh',
            'nagpur': 'Nagpur, Maharashtra',
            'indore': 'Indore, Madhya Pradesh',
            'thane': 'Thane, Maharashtra',
            'bhopal': 'Bhopal, Madhya Pradesh',
            'visakhapatnam': 'Visakhapatnam, Andhra Pradesh',
            'pimpri': 'Pimpri-Chinchwad, Maharashtra',
            'patna': 'Patna, Bihar',
            'vadodara': 'Vadodara, Gujarat',
            'ghaziabad': 'Ghaziabad, Uttar Pradesh',
            'ludhiana': 'Ludhiana, Punjab',
            'agra': 'Agra, Uttar Pradesh',
            'nashik': 'Nashik, Maharashtra',
            'faridabad': 'Faridabad, Haryana',
            'meerut': 'Meerut, Uttar Pradesh',
            'rajkot': 'Rajkot, Gujarat',
            'kalyan': 'Kalyan, Maharashtra',
            'vasai': 'Vasai-Virar, Maharashtra',
            'varanasi': 'Varanasi, Uttar Pradesh',
            'srinagar': 'Srinagar, Kashmir',
            'amritsar': 'Amritsar, Punjab',
            'aligarh': 'Aligarh, Uttar Pradesh',
            'guwahati': 'Guwahati, Assam',
            'chandigarh': 'Chandigarh',
            'jodhpur': 'Jodhpur, Rajasthan',
            'madurai': 'Madurai, Tamil Nadu',
            'raipur': 'Raipur, Chhattisgarh',
            'kota': 'Kota, Rajasthan',
            'gwalior': 'Gwalior, Madhya Pradesh',
            'vijayawada': 'Vijayawada, Andhra Pradesh',
            'mysore': 'Mysuru, Karnataka',
            'mysuru': 'Mysuru, Karnataka',
            'bhubaneswar': 'Bhubaneswar, Odisha',
            'salem': 'Salem, Tamil Nadu',
            'mira': 'Mira-Bhayandar, Maharashtra',
            'warangal': 'Warangal, Telangana',
            'thiruvananthapuram': 'Thiruvananthapuram, Kerala',
            'guntur': 'Guntur, Andhra Pradesh',
            'bhiwandi': 'Bhiwandi, Maharashtra',
            'saharanpur': 'Saharanpur, Uttar Pradesh',
            'gorakhpur': 'Gorakhpur, Uttar Pradesh',
            'bikaner': 'Bikaner, Rajasthan',
            'amravati': 'Amravati, Maharashtra',
            'noida': 'Noida, Uttar Pradesh',
            'jamshedpur': 'Jamshedpur, Jharkhand',
            'bhilai': 'Bhilai, Chhattisgarh',
            'cuttack': 'Cuttack, Odisha',
            'gulbarga': 'Gulbarga, Karnataka',
            'kochi': 'Kochi, Kerala',
            'udaipur': 'Udaipur, Rajasthan',
            'dehradun': 'Dehradun, Uttarakhand',
            'asansol': 'Asansol, West Bengal',
            'nanded': 'Nanded, Maharashtra',
            'kolhapur': 'Kolhapur, Maharashtra',
            'ajmer': 'Ajmer, Rajasthan',
            'akola': 'Akola, Maharashtra',
            'gulbarga': 'Gulbarga, Karnataka',
            'jamnagar': 'Jamnagar, Gujarat',
            'ujjain': 'Ujjain, Madhya Pradesh',
            'loni': 'Loni, Uttar Pradesh',
            'siliguri': 'Siliguri, West Bengal',
            'jhansi': 'Jhansi, Uttar Pradesh',
            'ulhasnagar': 'Ulhasnagar, Maharashtra',
            'nellore': 'Nellore, Andhra Pradesh',
            'jammu': 'Jammu, Jammu and Kashmir',
            'sangli': 'Sangli, Maharashtra',
            'mangalore': 'Mangalore, Karnataka',
            'erode': 'Erode, Tamil Nadu',
            'belgaum': 'Belgaum, Karnataka',
            'ambattur': 'Ambattur, Tamil Nadu',
            'tirunelveli': 'Tirunelveli, Tamil Nadu',
            'malegaon': 'Malegaon, Maharashtra',
            'gaya': 'Gaya, Bihar',
            'jalgaon': 'Jalgaon, Maharashtra',
            'udaipur': 'Udaipur, Rajasthan',
            'kozhikode': 'Kozhikode, Kerala',
            
            # Tourist destinations
            'goa': 'Panaji, Goa',
            'shimla': 'Shimla, Himachal Pradesh',
            'manali': 'Manali, Himachal Pradesh',
            'darjeeling': 'Darjeeling, West Bengal',
            'ooty': 'Udhagamandalam, Tamil Nadu',
            'udhagamandalam': 'Udhagamandalam, Tamil Nadu',
            'kodaikanal': 'Kodaikanal, Tamil Nadu',
            'munnar': 'Munnar, Kerala',
            'rishikesh': 'Rishikesh, Uttarakhand',
            'haridwar': 'Haridwar, Uttarakhand',
            'kedarnath': 'Kedarnath, Uttarakhand',
            'badrinath': 'Badrinath, Uttarakhand',
            'ladakh': 'Leh, Ladakh',
            'leh': 'Leh, Ladakh',
            'srinagar': 'Srinagar, Kashmir',
            'dharamshala': 'Dharamshala, Himachal Pradesh',
            'mcleodganj': 'McLeod Ganj, Himachal Pradesh',
            'mcleod ganj': 'McLeod Ganj, Himachal Pradesh',
            'pushkar': 'Pushkar, Rajasthan',
            'mount abu': 'Mount Abu, Rajasthan',
            'mountabu': 'Mount Abu, Rajasthan',
            'mt abu': 'Mount Abu, Rajasthan',
            'coorg': 'Coorg, Karnataka',
            'kodagu': 'Coorg, Karnataka',
            'hampi': 'Hampi, Karnataka',
            'gokarna': 'Gokarna, Karnataka',
            'varkala': 'Varkala, Kerala',
            'kovalam': 'Kovalam, Kerala',
            'alleppey': 'Alappuzha, Kerala',
            'alappuzha': 'Alappuzha, Kerala',
            'kumarakom': 'Kumarakom, Kerala',
            'wayanad': 'Wayanad, Kerala',
            'thekkady': 'Thekkady, Kerala',
            'ponmudi': 'Ponmudi, Kerala',
            'vagamon': 'Vagamon, Kerala',
            'mussoorie': 'Mussoorie, Uttarakhand',
            'nainital': 'Nainital, Uttarakhand',
            'almora': 'Almora, Uttarakhand',
            'ranikhet': 'Ranikhet, Uttarakhand',
            'kausani': 'Kausani, Uttarakhand',
            'auli': 'Auli, Uttarakhand',
            'jim corbett': 'Jim Corbett National Park, Uttarakhand',
            'corbett': 'Jim Corbett National Park, Uttarakhand',
            'ranthambore': 'Ranthambore National Park, Rajasthan',
            'khajuraho': 'Khajuraho, Madhya Pradesh',
            'orchha': 'Orchha, Madhya Pradesh',
            'pachmarhi': 'Pachmarhi, Madhya Pradesh',
            'mandu': 'Mandu, Madhya Pradesh',
            'chikmagalur': 'Chikmagalur, Karnataka',
            'chikkamagalur': 'Chikmagalur, Karnataka',
            'sakleshpur': 'Sakleshpur, Karnataka',
            'yercaud': 'Yercaud, Tamil Nadu',
            'coonoor': 'Coonoor, Tamil Nadu',
            'valparai': 'Valparai, Tamil Nadu',
            'yelagiri': 'Yelagiri, Tamil Nadu',
            'lonavala': 'Lonavala, Maharashtra',
            'khandala': 'Khandala, Maharashtra',
            'mahabaleshwar': 'Mahabaleshwar, Maharashtra',
            'panchgani': 'Panchgani, Maharashtra',
            'matheran': 'Matheran, Maharashtra',
            'karjat': 'Karjat, Maharashtra',
            'igatpuri': 'Igatpuri, Maharashtra',
            'kasauli': 'Kasauli, Himachal Pradesh',
            'chail': 'Chail, Himachal Pradesh',
            'kalka': 'Kalka, Haryana',
            'dalhousie': 'Dalhousie, Himachal Pradesh',
            'khajjiar': 'Khajjiar, Himachal Pradesh',
            'spiti': 'Spiti Valley, Himachal Pradesh',
            'kinnaur': 'Kinnaur, Himachal Pradesh',
            'kasol': 'Kasol, Himachal Pradesh',
            'tosh': 'Tosh, Himachal Pradesh',
            'malana': 'Malana, Himachal Pradesh',
            'bir': 'Bir, Himachal Pradesh',
            'billing': 'Billing, Himachal Pradesh',
            'chitkul': 'Chitkul, Himachal Pradesh',
            'kalpa': 'Kalpa, Himachal Pradesh',
            'sangla': 'Sangla, Himachal Pradesh',
            'tabo': 'Tabo, Himachal Pradesh',
            'kaza': 'Kaza, Himachal Pradesh',
            'chandratal': 'Chandratal Lake, Himachal Pradesh',
            'rohtang': 'Rohtang Pass, Himachal Pradesh',
            'solang': 'Solang Valley, Himachal Pradesh',
            'gulaba': 'Gulaba, Himachal Pradesh',
            'sissu': 'Sissu, Himachal Pradesh',
            'keylong': 'Keylong, Himachal Pradesh',
            'jispa': 'Jispa, Himachal Pradesh',
            'sarchu': 'Sarchu, Himachal Pradesh',
            'pangong': 'Pangong Lake, Ladakh',
            'nubra': 'Nubra Valley, Ladakh',
            'khardungla': 'Khardung La Pass, Ladakh',
            'tso moriri': 'Tso Moriri Lake, Ladakh',
            'zanskar': 'Zanskar Valley, Ladakh',
            'kargil': 'Kargil, Ladakh',
            'drass': 'Drass, Ladakh',
            'hemis': 'Hemis, Ladakh',
            'thiksey': 'Thiksey, Ladakh',
            'shey': 'Shey, Ladakh',
            'alchi': 'Alchi, Ladakh',
            'lamayuru': 'Lamayuru, Ladakh',
            'magnetic hill': 'Magnetic Hill, Ladakh',
            'gurudwara pathar sahib': 'Gurudwara Pathar Sahib, Ladakh',
            'hall of fame': 'Hall of Fame, Ladakh',
            'shanti stupa': 'Shanti Stupa, Ladakh',
            'leh palace': 'Leh Palace, Ladakh',
            'khardung la': 'Khardung La Pass, Ladakh',
            'diskit': 'Diskit, Ladakh',
            'hunder': 'Hunder, Ladakh',
            'panamik': 'Panamik, Ladakh',
            'sumur': 'Sumur, Ladakh',
            'turtuk': 'Turtuk, Ladakh',
            'chang la': 'Chang La Pass, Ladakh',
            'tanglang la': 'Tanglang La Pass, Ladakh',
            'nakula': 'Nakula Pass, Ladakh',
            'lachulung la': 'Lachulung La Pass, Ladakh',
            'more plains': 'More Plains, Ladakh',
            'pang': 'Pang, Ladakh',
        }
        
        # Common misspelling patterns
        self.common_corrections = {
            # Common misspellings and variations
            'hubili': 'hubli',
            'hubbali': 'hubli',
            'huballi': 'hubli',
            'bangalore': 'bengaluru',
            'bombay': 'mumbai',
            'calcutta': 'kolkata',
            'madras': 'chennai',
            'poona': 'pune',
            'mysore': 'mysuru',
            'cochin': 'kochi',
            'trivandrum': 'thiruvananthapuram',
            'pondicherry': 'puducherry',
            'vizag': 'visakhapatnam',
            
            # Tourist destinations with common misspellings
            'ladkh': 'ladakh',
            'ladak': 'ladakh',
            'ladahk': 'ladakh',
            'laddakh': 'ladakh',
            'leh ladkh': 'leh',
            'leh ladak': 'leh',
            
            'kedaranth': 'kedarnath',
            'kedranath': 'kedarnath',
            'kedarath': 'kedarnath',
            'kedarnth': 'kedarnath',
            
            'badrinath': 'badrinath',
            'badrinth': 'badrinath',
            'badriath': 'badrinath',
            
            'rishikesh': 'rishikesh',
            'rishiksh': 'rishikesh',
            'hrishikesh': 'rishikesh',
            
            'haridwar': 'haridwar',
            'hardwar': 'haridwar',
            'haridwr': 'haridwar',
            
            'manali': 'manali',
            'manli': 'manali',
            'manalli': 'manali',
            
            'shimla': 'shimla',
            'simla': 'shimla',
            'shimlla': 'shimla',
            
            'darjeeling': 'darjeeling',
            'darjiling': 'darjeeling',
            'darjelingg': 'darjeeling',
            
            'ooty': 'ooty',
            'ootty': 'ooty',
            'ootti': 'ooty',
            'udagamandalam': 'udhagamandalam',
            
            'kodaikanal': 'kodaikanal',
            'kodikanal': 'kodaikanal',
            'kodaiknal': 'kodaikanal',
            'kodikanel': 'kodaikanal',
            
            'munnar': 'munnar',
            'munar': 'munnar',
            'munner': 'munnar',
            
            # More corrections...
            'banaras': 'varanasi',
            'benares': 'varanasi',
            'kashi': 'varanasi',
            
            'jeypur': 'jaipur',
            'jaypur': 'jaipur',
            
            'udaypur': 'udaipur',
            'udaiperr': 'udaipur',
            
            'jodhperr': 'jodhpur',
            'jodhpurr': 'jodhpur',
            
            'pushkr': 'pushkar',
            'pushkarr': 'pushkar',
            
            'mcleodganj': 'mcleodganj',
            'mcleodgunj': 'mcleodganj',
            'mcleodgonj': 'mcleodganj',
            
            'dharamshalla': 'dharamshala',
            'dharmsala': 'dharamshala',
            'dharmshala': 'dharamshala',
            
            'varkalla': 'varkala',
            'varkla': 'varkala',
            
            'kovelam': 'kovalam',
            'kovlam': 'kovalam',
            
            'allepy': 'alleppey',
            
            'gokrna': 'gokarna',
            'hampe': 'hampi',
            'corg': 'coorg',
            'wyanad': 'wayanad',
            
            'masuri': 'mussoorie',
            'naini tal': 'nainital',
            'nainitel': 'nainital',
            
            'mountabu': 'mount abu',
            'mt abu': 'mount abu',
        }
        
        logger.info(f"🧠 Initialized location database with {len(self.known_locations)} known locations")
        logger.info(f"🔧 Loaded {len(self.common_corrections)} common correction patterns")

    def _smart_location_correction(self, location: str) -> Tuple[str, float]:
        """
        Intelligently correct location names using built-in pattern matching and fuzzy logic
        
        Args:
            location: Raw location input from user
            
        Returns:
            Tuple of (corrected_location, confidence_score)
        """
        original_location = location.strip()
        location_lower = original_location.lower()
        
        # Step 1: Direct match
        if location_lower in self.known_locations:
            return self.known_locations[location_lower], 1.0
        
        # Step 2: Check common correction patterns
        if location_lower in self.common_corrections:
            corrected_key = self.common_corrections[location_lower]
            if corrected_key in self.known_locations:
                corrected_location = self.known_locations[corrected_key]
                logger.info(f"🔧 Pattern correction: '{original_location}' → '{corrected_location}'")
                return corrected_location, 0.95
        
        # Step 3: Fuzzy matching using simple string similarity
        best_match = None
        best_score = 0.0
        
        for known_location in self.known_locations.keys():
            # Calculate similarity using a simple approach
            similarity = self._calculate_string_similarity(location_lower, known_location)
            if similarity > best_score:
                best_score = similarity
                best_match = known_location
        
        # Use fuzzy match if confidence is high enough
        if best_match and best_score >= 0.8:  # 80% similarity threshold
            corrected_location = self.known_locations[best_match]
            logger.info(f"🔧 Fuzzy match: '{original_location}' → '{corrected_location}' (similarity: {best_score:.2f})")
            return corrected_location, best_score
        
        # Step 4: Partial matching for compound locations
        for known_location in self.known_locations.keys():
            if len(location_lower) > 3 and location_lower in known_location:
                confidence = len(location_lower) / len(known_location)
                if confidence >= 0.5:
                    result = self.known_locations[known_location]
                    logger.info(f"🔧 Partial match: '{original_location}' → '{result}' (confidence: {confidence:.2f})")
                    return result, confidence
            elif len(known_location) > 3 and known_location in location_lower:
                confidence = len(known_location) / len(location_lower)
                if confidence >= 0.7:
                    result = self.known_locations[known_location]
                    logger.info(f"🔧 Reverse partial match: '{original_location}' → '{result}' (confidence: {confidence:.2f})")
                    return result, confidence
        
        # If no good correction found, return original with low confidence
        logger.warning(f"⚠️ Could not auto-correct location: '{original_location}'")
        return original_location, 0.1
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate string similarity using a simple character-based approach
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not str1 or not str2:
            return 0.0
        
        # Convert to lowercase and remove spaces
        s1 = str1.lower().replace(' ', '')
        s2 = str2.lower().replace(' ', '')
        
        # If strings are identical
        if s1 == s2:
            return 1.0
        
        # Calculate Levenshtein distance (simple implementation)
        distance = self._levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        
        if max_len == 0:
            return 1.0
        
        # Convert distance to similarity (0.0 to 1.0)
        similarity = 1.0 - (distance / max_len)
        return max(0.0, similarity)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Simple implementation of Levenshtein distance
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Edit distance between the strings
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def retry_api_call(self, func, *args, **kwargs):
        """
        Retry API calls with exponential backoff
        
        Args:
            func: Function to retry
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
            
        Raises:
            Exception: If all retry attempts fail
        """
        max_retries = 3
        base_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"✅ API call succeeded on attempt {attempt + 1}")
                return result
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"⚠️ API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    logger.info(f"🔄 Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"❌ API call failed after {max_retries} attempts: {str(e)}")
                    raise e

    def parse_query(self, query: str, current_date: str) -> Dict[str, Any]:
        """
        Parse user query to extract trip details with enhanced natural language support
        
        Args:
            query: User's trip planning query
            current_date: Current date in YYYY-MM-DD format
            
        Returns:
            Dictionary with parsed trip details
        """
        logger.info(f"🔍 Parsing query: '{query}'")
        
        # Default values
        parsed_data = {
            'location': 'Bangalore',
            'num_days': 3,
            'start_date': None,
            'end_date': None,
            'short_trip_warning': False
        }
        
        # Extract location (enhanced patterns to handle various natural language formats)
        location_patterns = [
            # Handle "want to go [to] [location]" patterns
            r'want\s+to\s+go\s+(?:to\s+)?([A-Za-z\s,]+?)(?:\s+(?:for|next|this|in|\d+)|\s*$)',
            r'want\s+to\s+go\s+for\s+([A-Za-z\s,]+?)(?:\s+(?:for|next|this|in|\d+)|\s*$)',
            # Handle "go to [location]" patterns  
            r'go\s+(?:for\s+trip\s+)?to\s+([A-Za-z\s,]+?)(?:\s+(?:next|this|in|for|\d+)|\s*$)',
            r'go\s+for\s+([A-Za-z\s,]+?)(?:\s+(?:for|next|this|in|\d+)|\s*$)',
            # Standard patterns
            r'to\s+([A-Za-z\s,]+?)(?:\s+(?:next|this|in|for|\d+)|\s*$)',
            r'visit\s+([A-Za-z\s,]+?)(?:\s+(?:next|this|in|for|\d+)|\s*$)',
            r'trip\s+to\s+([A-Za-z\s,]+?)(?:\s+(?:next|this|in|for|\d+)|\s*$)',
            r'in\s+([A-Za-z\s,]+?)(?:\s+(?:next|this|in|for|\d+)|\s*$)'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                parsed_data['location'] = match.group(1).strip()
                break
        
        # Extract number of days
        day_patterns = [
            r'(\d+)[-\s]days?',
            r'(\d+)[-\s]day',
            r'for\s+(\d+)\s+days?',
            r'(\d+)\s+day\s+trip'
        ]
        
        for pattern in day_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                parsed_data['num_days'] = int(match.group(1))
                break
        
        # Check for short trip warning
        if parsed_data['num_days'] < 2:
            parsed_data['short_trip_warning'] = True
            logger.warning(f"⚠️ Short trip detected: {parsed_data['num_days']} day(s)")
        
        # Parse current date
        try:
            current_dt = datetime.strptime(current_date, '%Y-%m-%d')
        except ValueError:
            current_dt = datetime.now()
        
        # Enhanced date parsing with more natural language support
        start_date = self._parse_enhanced_dates(query, current_dt)
        
        parsed_data['start_date'] = start_date.strftime('%Y-%m-%d')
        end_date = start_date + timedelta(days=parsed_data['num_days'] - 1)
        parsed_data['end_date'] = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"📋 Parsed details: {parsed_data}")
        return parsed_data
    
    def _parse_enhanced_dates(self, query: str, current_dt: datetime) -> datetime:
        """
        Enhanced date parsing for natural language inputs
        
        Args:
            query: User's query
            current_dt: Current datetime
            
        Returns:
            Parsed start date
        """
        query_lower = query.lower()
        
        # Handle specific date patterns
        # Look for specific dates like "August 20, 2024" or "20 August 2024"
        specific_date_patterns = [
            r'(\w+)\s+(\d{1,2}),?\s+(\d{4})',  # August 20, 2024
            r'(\d{1,2})\s+(\w+)\s+(\d{4})',   # 20 August 2024
            r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})',  # 20/08/2024 or 20-08-2024
            r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})'   # 2024-08-20
        ]
        
        for pattern in specific_date_patterns:
            match = re.search(pattern, query)
            if match:
                try:
                    if pattern == specific_date_patterns[0]:  # Month DD, YYYY
                        month_name, day, year = match.groups()
                        return datetime.strptime(f"{month_name} {day} {year}", "%B %d %Y")
                    elif pattern == specific_date_patterns[1]:  # DD Month YYYY
                        day, month_name, year = match.groups()
                        return datetime.strptime(f"{day} {month_name} {year}", "%d %B %Y")
                    elif pattern == specific_date_patterns[2]:  # DD/MM/YYYY
                        day, month, year = match.groups()
                        return datetime.strptime(f"{day}/{month}/{year}", "%d/%m/%Y")
                    elif pattern == specific_date_patterns[3]:  # YYYY-MM-DD
                        year, month, day = match.groups()
                        return datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")
                except ValueError:
                    continue
        
        # Handle relative dates
        if 'next monday' in query_lower:
            days_ahead = (0 - current_dt.weekday()) % 7 + 7  # Next Monday
            return current_dt + timedelta(days=days_ahead)
        elif 'next tuesday' in query_lower:
            days_ahead = (1 - current_dt.weekday()) % 7 + 7
            return current_dt + timedelta(days=days_ahead)
        elif 'next wednesday' in query_lower:
            days_ahead = (2 - current_dt.weekday()) % 7 + 7
            return current_dt + timedelta(days=days_ahead)
        elif 'next thursday' in query_lower:
            days_ahead = (3 - current_dt.weekday()) % 7 + 7
            return current_dt + timedelta(days=days_ahead)
        elif 'next friday' in query_lower:
            days_ahead = (4 - current_dt.weekday()) % 7 + 7
            return current_dt + timedelta(days=days_ahead)
        elif 'next saturday' in query_lower:
            days_ahead = (5 - current_dt.weekday()) % 7 + 7
            return current_dt + timedelta(days=days_ahead)
        elif 'next sunday' in query_lower:
            days_ahead = (6 - current_dt.weekday()) % 7 + 7
            return current_dt + timedelta(days=days_ahead)
        elif 'next weekend' in query_lower:
            # Next Saturday
            days_ahead = (5 - current_dt.weekday()) % 7 + 7
            return current_dt + timedelta(days=days_ahead)
        elif 'this weekend' in query_lower:
            # This Saturday
            days_ahead = (5 - current_dt.weekday()) % 7
            if days_ahead == 0 and current_dt.weekday() == 5:  # If today is Saturday
                return current_dt
            return current_dt + timedelta(days=days_ahead)
        elif 'next week' in query_lower:
            return current_dt + timedelta(days=7)
        elif 'this week' in query_lower:
            return current_dt + timedelta(days=1)
        elif 'next month' in query_lower:
            return current_dt + timedelta(days=30)
        elif 'tomorrow' in query_lower:
            return current_dt + timedelta(days=1)
        elif 'day after tomorrow' in query_lower:
            return current_dt + timedelta(days=2)
        elif re.search(r'next\s+(\d+)\s+days?', query_lower):
            # Handle "next X days" - start from tomorrow
            return current_dt + timedelta(days=1)
        else:
            # Default to next week if no date specified
            return current_dt + timedelta(days=7)

    def _normalize_location_name(self, location: str) -> List[str]:
        """
        Normalize and generate variations of location names using intelligent auto-correction
        
        Args:
            location: Raw location name from user input
            
        Returns:
            List of normalized location variations to try
        """
        # Use intelligent auto-correction
        corrected_location, confidence = self._smart_location_correction(location)
        
        variations = []
        
        # If we have high confidence in the correction, prioritize it
        if confidence >= 0.8:
            variations.append(corrected_location)
            logger.info(f"🎯 High confidence auto-correction: '{location}' → '{corrected_location}' (confidence: {confidence:.2f})")
        
        # Always include original variations for fallback
        variations.extend([
            location.title().strip(),
            location.strip(),
            f"{location.strip()}, India"
        ])
        
        # Add corrected location variations if different from original
        if corrected_location.lower() != location.lower():
            variations.extend([
                corrected_location,
                f"{corrected_location}, India"
            ])
        
        # Add state-specific variations for major cities
        city_lower = corrected_location.lower().split(',')[0].strip()
        state_mappings = {
            'mumbai': 'Mumbai, Maharashtra, India',
            'delhi': 'New Delhi, Delhi, India',
            'bengaluru': 'Bengaluru, Karnataka, India',
            'kolkata': 'Kolkata, West Bengal, India',
            'chennai': 'Chennai, Tamil Nadu, India',
            'hyderabad': 'Hyderabad, Telangana, India',
            'pune': 'Pune, Maharashtra, India',
            'jaipur': 'Jaipur, Rajasthan, India',
            'kochi': 'Kochi, Kerala, India',
            'goa': 'Panaji, Goa, India',
            'shimla': 'Shimla, Himachal Pradesh, India',
            'manali': 'Manali, Himachal Pradesh, India',
            'darjeeling': 'Darjeeling, West Bengal, India',
            'ooty': 'Udhagamandalam, Tamil Nadu, India',
            'munnar': 'Munnar, Kerala, India',
            'ladakh': 'Leh, Ladakh, India',
            'kedarnath': 'Kedarnath, Uttarakhand, India',
            'rishikesh': 'Rishikesh, Uttarakhand, India'
        }
        
        if city_lower in state_mappings:
            variations.append(state_mappings[city_lower])
        
        # Remove duplicates while preserving order
        unique_variations = []
        for var in variations:
            if var and var not in unique_variations:
                unique_variations.append(var)
        
        logger.info(f"📍 Generated {len(unique_variations)} location variations for '{location}'")
        if confidence >= 0.8:
            logger.info(f"🔧 Auto-correction applied: '{location}' → '{corrected_location}'")
        
        return unique_variations

    def geocode_location(self, address: str, api_key: str) -> Tuple[float, float]:
        """
        Get coordinates for a location using Geoapify API with enhanced location matching
        
        Args:
            address: Location name or address
            api_key: Geoapify API key
            
        Returns:
            Tuple of (latitude, longitude)
            
        Raises:
            Exception: If geocoding fails for all variations
        """
        # Get location variations to try
        location_variations = self._normalize_location_name(address)
        
        logger.info(f"🌍 Geocoding location: {address}")
        logger.info(f"🔍 Trying variations: {location_variations}")
        
        # Try each variation
        for i, variation in enumerate(location_variations):
            try:
                def _make_geocode_request():
                    url = "https://api.geoapify.com/v1/geocode/search"
                    params = {
                        'text': variation,
                        'apiKey': api_key,
                        'limit': 1
                    }
                    
                    logger.info(f"🔍 Trying geocoding variation {i+1}/{len(location_variations)}: {variation}")
                    response = requests.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    
                    data = response.json()
                    if not data.get('features'):
                        raise Exception(f"No geocoding results found for '{variation}'")
                    
                    coordinates = data['features'][0]['geometry']['coordinates']
                    lat, lon = coordinates[1], coordinates[0]  # GeoJSON uses [lon, lat]
                    
                    logger.info(f"✅ Successfully geocoded '{variation}' to coordinates: ({lat}, {lon})")
                    return lat, lon
                
                # Try this variation with retry logic
                result = self.retry_api_call(_make_geocode_request)
                return result
                        
            except Exception as e:
                logger.warning(f"⚠️ Variation '{variation}' failed: {str(e)}")
                continue
        
        # If all variations failed, raise exception
        raise Exception(f"No geocoding results found for '{address}' or any of its variations: {location_variations}")

    def get_weather(self, lat: float, lon: float, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Get weather forecast using Open-Meteo API (free, no key required)
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with weather summary and daily forecasts
            
        Raises:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with weather summary and daily forecasts
            
        Raises:
            Exception: If weather data retrieval fails
        """
        def _make_weather_request():
            # Check if dates are too far in the future (Open-Meteo has limitations)
            today = datetime.now().date()
            start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
            days_ahead = (start_dt - today).days
            
            # Open-Meteo typically supports up to 16 days forecast
            if days_ahead > 16:
                logger.warning(f"⚠️ Weather request for {days_ahead} days ahead, using fallback weather data")
                # Use fallback weather data for distant future dates
                return self._generate_fallback_weather(start_date, end_date, lat, lon)
            
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                'latitude': lat,
                'longitude': lon,
                'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code',
                'start_date': start_date,
                'end_date': end_date,
                'timezone': 'auto'
            }
            
            logger.info(f"🌤️ Fetching weather for coordinates ({lat}, {lon}) from {start_date} to {end_date}")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            daily_data = data.get('daily', {})
            
            # Process weather data
            weather_summary = {
                'daily_forecasts': [],
                'overall_conditions': 'moderate',
                'outdoor_friendly_days': 0
            }
            
            dates = daily_data.get('time', [])
            max_temps = daily_data.get('temperature_2m_max', [])
            min_temps = daily_data.get('temperature_2m_min', [])
            precipitation = daily_data.get('precipitation_sum', [])
            weather_codes = daily_data.get('weather_code', [])
            
            for i, date in enumerate(dates):
                max_temp = max_temps[i] if i < len(max_temps) else 20
                min_temp = min_temps[i] if i < len(min_temps) else 15
                precip = precipitation[i] if i < len(precipitation) else 0
                code = weather_codes[i] if i < len(weather_codes) else 0
                
                # Determine weather condition
                condition = self._get_weather_condition(code)
                
                # Check if suitable for outdoor activities (15-25°C, low precipitation)
                outdoor_friendly = (15 <= max_temp <= 25) and (precip < 5)
                if outdoor_friendly:
                    weather_summary['outdoor_friendly_days'] += 1
                
                daily_forecast = {
                    'date': date,
                    'max_temp': round(max_temp, 1),
                    'min_temp': round(min_temp, 1),
                    'precipitation': round(precip, 1),
                    'condition': condition,
                    'outdoor_friendly': outdoor_friendly
                }
                weather_summary['daily_forecasts'].append(daily_forecast)
            
            # Determine overall conditions
            if weather_summary['outdoor_friendly_days'] >= len(dates) * 0.7:
                weather_summary['overall_conditions'] = 'excellent'
            elif weather_summary['outdoor_friendly_days'] >= len(dates) * 0.4:
                weather_summary['overall_conditions'] = 'good'
            else:
                weather_summary['overall_conditions'] = 'challenging'
            
            logger.info(f"☀️ Weather summary: {weather_summary['overall_conditions']} conditions, "
                       f"{weather_summary['outdoor_friendly_days']} outdoor-friendly days")
            
            return weather_summary
        
        return self.retry_api_call(_make_weather_request)

    def _generate_fallback_weather(self, start_date: str, end_date: str, lat: float, lon: float) -> Dict[str, Any]:
        """
        Generate fallback weather data for dates beyond forecast range
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            lat: Latitude for regional weather patterns
            lon: Longitude for regional weather patterns
            
        Returns:
            Fallback weather summary
        """
        logger.info(f"🌤️ Generating fallback weather data for {start_date} to {end_date}")
        
        # Generate dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        current_dt = start_dt
        
        daily_forecasts = []
        outdoor_friendly_days = 0
        
        # Regional weather patterns based on location (basic approximation)
        # For Indian locations, consider seasonal patterns
        month = start_dt.month
        
        # Basic seasonal weather for India
        if month in [12, 1, 2]:  # Winter
            base_max_temp, base_min_temp = 22, 12
            base_precip = 1
            base_condition = "Clear sky"
        elif month in [3, 4, 5]:  # Summer
            base_max_temp, base_min_temp = 32, 20
            base_precip = 2
            base_condition = "Partly cloudy"
        elif month in [6, 7, 8, 9]:  # Monsoon
            base_max_temp, base_min_temp = 28, 22
            base_precip = 8
            base_condition = "Rain"
        else:  # Post-monsoon
            base_max_temp, base_min_temp = 26, 18
            base_precip = 3
            base_condition = "Partly cloudy"
        
        while current_dt <= end_dt:
            # Add some variation to make it realistic
            max_temp = base_max_temp + (hash(str(current_dt.day)) % 8 - 4)
            min_temp = base_min_temp + (hash(str(current_dt.day)) % 6 - 3)
            precip = max(0, base_precip + (hash(str(current_dt.day)) % 6 - 3))
            
            # Ensure min < max
            if min_temp >= max_temp:
                min_temp = max_temp - 5
            
            outdoor_friendly = (15 <= max_temp <= 30) and (precip < 5)
            if outdoor_friendly:
                outdoor_friendly_days += 1
            
            daily_forecast = {
                'date': current_dt.strftime('%Y-%m-%d'),
                'max_temp': round(max_temp, 1),
                'min_temp': round(min_temp, 1),
                'precipitation': round(precip, 1),
                'condition': base_condition,
                'outdoor_friendly': outdoor_friendly
            }
            daily_forecasts.append(daily_forecast)
            current_dt += timedelta(days=1)
        
        # Determine overall conditions
        total_days = len(daily_forecasts)
        if outdoor_friendly_days >= total_days * 0.7:
            overall_conditions = 'excellent'
        elif outdoor_friendly_days >= total_days * 0.4:
            overall_conditions = 'good'
        else:
            overall_conditions = 'challenging'
        
        weather_summary = {
            'daily_forecasts': daily_forecasts,
            'overall_conditions': overall_conditions,
            'outdoor_friendly_days': outdoor_friendly_days
        }
        
        logger.info(f"📅 Generated fallback weather: {overall_conditions} conditions, {outdoor_friendly_days} outdoor-friendly days")
        return weather_summary

    def _get_weather_condition(self, weather_code: int) -> str:
        """Convert weather code to readable condition"""
        if weather_code == 0:
            return "Clear sky"
        elif weather_code in [1, 2, 3]:
            return "Partly cloudy"
        elif weather_code in [45, 48]:
            return "Foggy"
        elif weather_code in [51, 53, 55]:
            return "Drizzle"
        elif weather_code in [61, 63, 65]:
            return "Rain"
        elif weather_code in [71, 73, 75]:
            return "Snow"
        elif weather_code in [95, 96, 99]:
            return "Thunderstorm"
        else:
            return "Partly cloudy"

    def get_places(self, lat: float, lon: float, weather_summary: Dict, num_days: int, api_key: str, location_name: str = "Unknown City") -> List[Dict[str, Any]]:
        """
        Get places of interest using Geoapify API with enhanced diversity and multiple calls
        
        Args:
            lat: Latitude
            lon: Longitude
            weather_summary: Weather forecast data
            num_days: Number of trip days
            api_key: Geoapify API key
            location_name: Name of the destination location
            
        Returns:
            List of place dictionaries with enhanced information
            
        Raises:
            Exception: If places retrieval fails
        """
        def _make_places_request():
            # Comprehensive list of place categories for better diversity
            outdoor_days = weather_summary.get('outdoor_friendly_days', 0)
            total_days = len(weather_summary.get('daily_forecasts', []))
            
            # Base categories for all conditions
            base_categories = [
                'tourism.attraction',
                'entertainment.museum',
                'catering.restaurant',
                'heritage.monument',
                'entertainment.culture',
                'leisure.marina'
            ]
            
            # Weather-dependent categories
            if outdoor_days >= total_days * 0.7:
                # Good weather - add outdoor categories
                additional_categories = [
                    'leisure.park',
                    'leisure.garden',
                    'sport.climbing',
                    'tourism.viewpoint',
                    'leisure.beach_resort',
                    'natural.beach'
                ]
            elif outdoor_days >= total_days * 0.4:
                # Mixed weather - balanced indoor/outdoor
                additional_categories = [
                    'commercial.shopping_mall',
                    'leisure.park',
                    'entertainment.cinema',
                    'tourism.zoo'
                ]
            else:
                # Poor weather - focus on indoor activities
                additional_categories = [
                    'commercial.shopping_mall',
                    'entertainment.cinema',
                    'entertainment.arts_centre',
                    'tourism.gallery'
                ]
            
            # Combine categories
            all_categories = base_categories + additional_categories
            all_places = []
            
            # Make multiple API calls for better diversity
            places_per_category = max(3, (num_days * 4) // len(all_categories))
            
            for category in all_categories:
                try:
                    url = "https://api.geoapify.com/v2/places"
                    params = {
                        'categories': category,
                        'filter': f'circle:{lon},{lat},15000',  # Increased radius to 15km
                        'bias': f'proximity:{lon},{lat}',
                        'limit': places_per_category,
                        'apiKey': api_key
                    }
                    
                    logger.info(f"🎯 Searching for places in category: {category}")
                    response = requests.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    
                    data = response.json()
                    features = data.get('features', [])
                    
                    for feature in features:
                        properties = feature.get('properties', {})
                        coordinates = feature.get('geometry', {}).get('coordinates', [0, 0])
                        
                        # Extract additional information
                        rating = properties.get('rating', 0)
                        opening_hours = properties.get('opening_hours', 'Hours not available')
                        website = properties.get('website', '')
                        phone = properties.get('phone', '')
                        
                        # Create description based on category and properties
                        description = self._generate_place_description(properties, category)
                        
                        # Create location-specific fallback names
                        category_key = category.split('.')[-1].lower()
                        
                        # Extract the actual place name and location details for better identification
                        street = properties.get('street', '')
                        address_line1 = properties.get('address_line1', '')
                        district = properties.get('district', '')
                        city = properties.get('city', '')
                        specific_location = ' '.join(filter(None, [street, district, city])).strip()
                        
                        if not specific_location and address_line1:
                            specific_location = address_line1
                            
                        fallback_names = {
                            'attraction': f'Popular Attraction at {specific_location if specific_location else location_name}',
                            'museum': f'Local Museum at {specific_location if specific_location else location_name}',
                            'restaurant': f'Local Restaurant at {specific_location if specific_location else location_name}',
                            'monument': f'Historic Monument at {specific_location if specific_location else location_name}',
                            'park': f'City Park at {specific_location if specific_location else location_name}',
                            'garden': f'Botanical Garden at {specific_location if specific_location else location_name}',
                            'viewpoint': f'Scenic Viewpoint at {specific_location if specific_location else location_name}',
                            'beach': f'Beach at {specific_location if specific_location else location_name}',
                            'shopping_mall': f'Shopping Center at {specific_location if specific_location else location_name}',
                            'cinema': f'Movie Theater at {specific_location if specific_location else location_name}',
                            'gallery': f'Art Gallery at {specific_location if specific_location else location_name}',
                            'zoo': f'Zoo at {specific_location if specific_location else location_name}',
                            'culture': f'Cultural Center at {specific_location if specific_location else location_name}',
                            'marina': f'Marina at {specific_location if specific_location else location_name}',
                            'climbing': f'Adventure Sports at {specific_location if specific_location else location_name}',
                            'beach_resort': f'Beach Resort at {specific_location if specific_location else location_name}'
                        }
                        fallback_name = fallback_names.get(category_key, f'Interesting Place at {specific_location if specific_location else location_name}')
                        
                        # Use actual name if available, otherwise use enhanced fallback name with specific location
                        place_name = properties.get('name')
                        if not place_name:
                            place_name = fallback_name
                        # If we have a name but it's very generic, enhance it with the location
                        elif len(place_name) < 10 and specific_location:
                            place_name = f"{place_name} ({specific_location})"
                            
                        place = {
                            'name': place_name,
                            'category': category.split('.')[-1].title(),
                            'address': properties.get('formatted', 'Address not available'),
                            'lat': coordinates[1],
                            'lon': coordinates[0],
                            'distance_km': self._calculate_distance(lat, lon, coordinates[1], coordinates[0]),
                            'rating': rating,
                            'opening_hours': opening_hours,
                            'website': website,
                            'phone': phone,
                            'description': description,
                            'relevance_score': self._calculate_relevance_score(properties, category, weather_summary)
                        }
                        all_places.append(place)
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to fetch places for category {category}: {str(e)}")
                    continue
            
            # Sort by relevance score (if available), then by rating, then by distance
            all_places.sort(key=lambda x: (-x['relevance_score'], -x['rating'], x['distance_km']))
            
            # Remove duplicates based on name and location
            unique_places = []
            seen_places = set()
            
            for place in all_places:
                place_key = (place['name'].lower(), round(place['lat'], 4), round(place['lon'], 4))
                if place_key not in seen_places:
                    unique_places.append(place)
                    seen_places.add(place_key)
            
            # Limit total places but ensure good variety
            max_places = num_days * 4  # 4 places per day maximum for better choice
            selected_places = unique_places[:max_places]
            
            logger.info(f"📍 Found {len(selected_places)} unique places for {num_days}-day trip")
            logger.info(f"🏆 Top rated place: {selected_places[0]['name'] if selected_places else 'None'}")
            
            return selected_places
        
        return self.retry_api_call(_make_places_request)
    
    def _generate_place_description(self, properties: Dict, category: str) -> str:
        """Generate descriptive text for a place based on its properties and category"""
        category_descriptions = {
            'attraction': "A popular tourist destination offering unique experiences and sightseeing opportunities.",
            'museum': "A cultural institution showcasing art, history, or science collections for educational visits.",
            'restaurant': "A dining establishment offering local or international cuisine in a welcoming atmosphere.",
            'monument': "A significant historical or cultural landmark representing the area's heritage and stories.",
            'park': "A green space perfect for relaxation, outdoor activities, and enjoying nature.",
            'garden': "A beautifully landscaped area ideal for peaceful walks and botanical exploration.",
            'viewpoint': "A scenic location offering panoramic views and excellent photo opportunities.",
            'beach': "A coastal area perfect for water activities, relaxation, and beachside recreation.",
            'shopping_mall': "A retail complex with various shops, dining options, and entertainment facilities.",
            'cinema': "An entertainment venue showing the latest movies in comfortable modern facilities.",
            'gallery': "An art space featuring exhibitions and collections from local and international artists.",
            'zoo': "A wildlife park where visitors can observe and learn about various animal species."
        }
        
        category_key = category.split('.')[-1].lower()
        base_description = category_descriptions.get(category_key, "An interesting place worth visiting during your trip.")
        
        # Add specific details if available
        details = []
        if properties.get('rating') and properties.get('rating') > 4:
            details.append("highly rated by visitors")
        if 'historic' in properties.get('name', '').lower():
            details.append("with historical significance")
        if any(word in properties.get('name', '').lower() for word in ['royal', 'palace', 'fort']):
            details.append("featuring royal architecture")
        
        if details:
            base_description += f" This location is {', '.join(details)}."
        
        return base_description
    
    def _calculate_relevance_score(self, properties: Dict, category: str, weather_summary: Dict) -> float:
        """Calculate a relevance score for a place based on various factors"""
        score = 0.0
        
        # Base score from rating
        rating = properties.get('rating', 0)
        if rating > 0:
            score += rating * 2  # Rating contributes 0-10 points
        
        # Category relevance based on weather
        outdoor_friendly_days = weather_summary.get('outdoor_friendly_days', 0)
        total_days = len(weather_summary.get('daily_forecasts', []))
        outdoor_ratio = outdoor_friendly_days / max(total_days, 1)
        
        outdoor_categories = ['park', 'garden', 'viewpoint', 'beach', 'climbing']
        indoor_categories = ['museum', 'shopping_mall', 'cinema', 'gallery']
        
        category_name = category.split('.')[-1].lower()
        
        if category_name in outdoor_categories:
            score += outdoor_ratio * 5  # More points for outdoor places in good weather
        elif category_name in indoor_categories:
            score += (1 - outdoor_ratio) * 5  # More points for indoor places in bad weather
        else:
            score += 2.5  # Neutral categories get medium points
        
        # Bonus for popular attractions
        if 'attraction' in category or 'monument' in category:
            score += 3
        
        # Bonus for restaurants (always needed)
        if 'restaurant' in category or 'catering' in category:
            score += 2
        
        return score

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        from math import radians, cos, sin, asin, sqrt
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Earth's radius in kilometers
        
        return round(c * r, 2)
    
    def _format_duration(self, duration_minutes: int) -> str:
        """Format duration for better display"""
        if duration_minutes < 60:
            return f"{duration_minutes} minutes"
        else:
            hours = duration_minutes // 60
            remaining_minutes = duration_minutes % 60
            if remaining_minutes == 0:
                return f"{hours} hour{'s' if hours != 1 else ''}"
            else:
                return f"{hours}h {remaining_minutes}m"

    def get_route(self, start_lat: float, start_lon: float, end_lat: float, end_lon: float, api_key: str) -> Dict[str, Any]:
        """
        Get route information using OpenRouteService API
        
        Args:
            start_lat: Starting latitude
            start_lon: Starting longitude
            end_lat: Destination latitude
            end_lon: Destination longitude
            api_key: OpenRouteService API key
            
        Returns:
            Dictionary with route details
            
        Raises:
            Exception: If route calculation fails
        """
        def _make_route_request():
            url = "https://api.openrouteservice.org/v2/directions/driving-car"
            headers = {
                'Authorization': api_key,
                'Content-Type': 'application/json'
            }
            
            body = {
                'coordinates': [[start_lon, start_lat], [end_lon, end_lat]],
                'format': 'json'
            }
            
            logger.info(f"🛣️ Calculating route from ({start_lat}, {start_lon}) to ({end_lat}, {end_lon})")
            response = requests.post(url, headers=headers, json=body, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            route = data.get('routes', [{}])[0]
            summary = route.get('summary', {})
            
            route_info = {
                'distance_km': round(summary.get('distance', 0) / 1000, 2),
                'duration_minutes': round(summary.get('duration', 0) / 60, 0),
                'duration_hours': round(summary.get('duration', 0) / 3600, 1)
            }
            
            # Format duration for better display
            duration_minutes = route_info['duration_minutes']
            if duration_minutes < 60:
                route_info['duration_display'] = f"{duration_minutes} minutes"
            else:
                hours = duration_minutes // 60
                remaining_minutes = duration_minutes % 60
                if remaining_minutes == 0:
                    route_info['duration_display'] = f"{hours} hour{'s' if hours != 1 else ''}"
                else:
                    route_info['duration_display'] = f"{hours}h {remaining_minutes}m"
            
            logger.info(f"🚗 Route: {route_info['distance_km']}km, {route_info['duration_display']}")
            return route_info
        
        return self.retry_api_call(_make_route_request)

    def generate_itinerary(self, parsed_data: Dict, weather: Dict, places: List[Dict], routes: List[Dict]) -> str:
        """
        Generate a formatted Markdown itinerary with enhanced routing and descriptions
        
        Args:
            parsed_data: Parsed query data
            weather: Weather forecast data
            places: List of places to visit
            routes: List of route information
            
        Returns:
            Formatted Markdown itinerary string
        """
        logger.info("📝 Generating complete itinerary")
        
        location = parsed_data['location']
        num_days = parsed_data['num_days']
        start_date = parsed_data['start_date']
        short_trip_warning = parsed_data.get('short_trip_warning', False)
        
        # Start building the itinerary
        itinerary = f"# 🌴 {num_days}-Day Trip to {location}\n\n"
        itinerary += f"**Trip Dates:** {start_date} to {parsed_data['end_date']}\n\n"
        
        # Short trip warning
        if short_trip_warning:
            itinerary += "⚠️ **Short Trip Notice:** This is a very short trip duration. Consider extending your stay to fully experience the destination.\n\n"
        
        # High-altitude acclimatization tip
        high_altitude_locations = ['ladakh', 'leh', 'manali', 'shimla', 'mcleodganj', 'dharamshala', 'spiti', 'kashmir', 'tibet', 'nepal', 'bhutan']
        if any(alt_loc in location.lower() for alt_loc in high_altitude_locations):
            itinerary += "## 🏔️ Day 1 Acclimatization Notice\n\n"
            itinerary += "**Important:** This destination is at high altitude. On Day 1, take it easy to acclimatize:\n"
            itinerary += "- Avoid strenuous activities for the first 24 hours\n"
            itinerary += "- Drink plenty of water and avoid alcohol\n"
            itinerary += "- Rest frequently and listen to your body\n"
            itinerary += "- Consider light walks instead of intensive sightseeing\n"
            itinerary += "- Consult a doctor if you experience severe headaches, nausea, or breathing difficulties\n\n"
        
        # Weather Summary
        itinerary += "## 🌤️ Weather Overview\n\n"
        conditions = weather.get('overall_conditions', 'moderate')
        outdoor_days = weather.get('outdoor_friendly_days', 0)
        itinerary += f"**Overall Conditions:** {conditions.title()}\n"
        itinerary += f"**Outdoor-Friendly Days:** {outdoor_days} out of {num_days}\n\n"
        
        # Daily weather breakdown
        itinerary += "### Daily Weather Forecast\n\n"
        for forecast in weather.get('daily_forecasts', []):
            date = forecast['date']
            condition = forecast['condition']
            max_temp = forecast['max_temp']
            min_temp = forecast['min_temp']
            precip = forecast['precipitation']
            outdoor = "✅" if forecast['outdoor_friendly'] else "⚠️"
            
            # Enhanced precipitation symbols based on amount
            if precip == 0:
                precip_symbol = "☀️"
            elif precip <= 2:
                precip_symbol = "🌤️"
            elif precip <= 5:
                precip_symbol = "🌦️"
            elif precip <= 10:
                precip_symbol = "🌧️"
            else:
                precip_symbol = "⛈️"
            
            itinerary += f"- **{date}**: {condition}, {min_temp}°C - {max_temp}°C, "
            itinerary += f"Precipitation: {precip}mm {precip_symbol}\n"
        
        # Recommended Places Overview
        if places:
            itinerary += "\n## 🎯 Recommended Places Overview\n\n"
            for i, place in enumerate(places[:6], 1):  # Show top 6 places
                itinerary += f"### {i}. {place['name']} ({place['category']})\n"
                itinerary += f"📍 **Location:** {place['address']}\n"
                itinerary += f"📏 **Distance:** {place['distance_km']}km from city center\n"
                if place.get('rating') and place['rating'] > 0:
                    itinerary += f"⭐ **Rating:** {place['rating']}/5\n"
                if place.get('description'):
                    itinerary += f"📝 **Description:** {place['description']}\n"
                if place.get('opening_hours') and place['opening_hours'] != 'Hours not available':
                    itinerary += f"🕐 **Hours:** {place['opening_hours']}\n"
                itinerary += "\n"
        
        # Daily Itinerary with Enhanced Routing
        itinerary += "## 📅 Daily Itinerary with Routes\n\n"
        
        places_per_day = len(places) // num_days if places else 1
        places_index = 0
        
        for day in range(num_days):
            day_num = day + 1
            current_date = weather['daily_forecasts'][day] if day < len(weather['daily_forecasts']) else None
            
            itinerary += f"### Day {day_num}"
            if current_date:
                itinerary += f" - {current_date['date']}"
            itinerary += "\n\n"
            
            if current_date:
                itinerary += f"**Weather:** {current_date['condition']}, "
                itinerary += f"{current_date['min_temp']}°C - {current_date['max_temp']}°C\n\n"
            
            # Get places for this day
            daily_places = []
            places_for_day = min(3, len(places) - places_index)
            for _ in range(places_for_day):
                if places_index < len(places):
                    daily_places.append(places[places_index])
                    places_index += 1
            
            # Generate routes for daily places
            daily_routes = []
            if len(daily_places) >= 2:
                try:
                    # Calculate routes between consecutive places
                    for i in range(len(daily_places) - 1):
                        route = self.get_route(
                            daily_places[i]['lat'], daily_places[i]['lon'],
                            daily_places[i + 1]['lat'], daily_places[i + 1]['lon'],
                            self.ors_api_key
                        )
                        route['from'] = daily_places[i]['name']
                        route['to'] = daily_places[i + 1]['name']
                        daily_routes.append(route)
                except Exception as e:
                    logger.warning(f"⚠️ Route calculation failed for day {day_num}: {str(e)}")
            
            # Morning activity
            itinerary += "**Morning (9:00 AM - 12:00 PM)**\n"
            if len(daily_places) > 0:
                place = daily_places[0]
                itinerary += f"- Visit **{place['name']}** ({place['category']})\n"
                itinerary += f"  - 📍 Address: {place['address']}\n"
                itinerary += f"  - 📏 Distance from center: {place['distance_km']}km\n"
                if place.get('description'):
                    itinerary += f"  - 📝 {place['description']}\n"
                if place.get('rating') and place['rating'] > 0:
                    itinerary += f"  - ⭐ Rating: {place['rating']}/5\n"
            else:
                itinerary += "- Explore local markets and street food\n"
            
            # Route to afternoon location
            if len(daily_routes) > 0:
                route = daily_routes[0]
                itinerary += f"\n🚗 **Route to next location:**\n"
                itinerary += f"- From: {route['from']}\n"
                itinerary += f"- To: {route['to']}\n"
                itinerary += f"- Distance: {route['distance_km']}km\n"
                itinerary += f"- Travel time: {self._format_duration(int(route['duration_minutes']))}\n"
            
            # Afternoon activity
            itinerary += "\n**Afternoon (1:00 PM - 5:00 PM)**\n"
            if len(daily_places) > 1:
                place = daily_places[1]
                itinerary += f"- Explore **{place['name']}** ({place['category']})\n"
                itinerary += f"  - 📍 Address: {place['address']}\n"
                itinerary += f"  - 📏 Distance from center: {place['distance_km']}km\n"
                if place.get('description'):
                    itinerary += f"  - 📝 {place['description']}\n"
                if place.get('rating') and place['rating'] > 0:
                    itinerary += f"  - ⭐ Rating: {place['rating']}/5\n"
            else:
                itinerary += "- Rest and relaxation time\n"
            
            # Route to evening location
            if len(daily_routes) > 1:
                route = daily_routes[1]
                itinerary += f"\n🚗 **Route to evening location:**\n"
                itinerary += f"- From: {route['from']}\n"
                itinerary += f"- To: {route['to']}\n"
                itinerary += f"- Distance: {route['distance_km']}km\n"
                itinerary += f"- Travel time: {self._format_duration(int(route['duration_minutes']))}\n"
            
            # Evening activity
            itinerary += "\n**Evening (6:00 PM - 9:00 PM)**\n"
            if len(daily_places) > 2:
                place = daily_places[2]
                itinerary += f"- Dinner at **{place['name']}** ({place['category']})\n"
                itinerary += f"  - 📍 Address: {place['address']}\n"
                itinerary += f"  - 📏 Distance from center: {place['distance_km']}km\n"
                if place.get('description'):
                    itinerary += f"  - 📝 {place['description']}\n"
                if place.get('rating') and place['rating'] > 0:
                    itinerary += f"  - ⭐ Rating: {place['rating']}/5\n"
            else:
                itinerary += "- Local dining experience\n"
            
            # Daily travel summary
            if daily_routes:
                total_distance = sum(route['distance_km'] for route in daily_routes)
                total_time = sum(route['duration_minutes'] for route in daily_routes)
                itinerary += f"\n📊 **Daily Travel Summary:**\n"
                itinerary += f"- Total distance: {total_distance:.1f}km\n"
                itinerary += f"- Total travel time: {self._format_duration(int(total_time))}\n"
            
            # Weather-specific recommendations
            if current_date:
                if not current_date['outdoor_friendly']:
                    itinerary += "\n**💡 Weather Tip:** Consider indoor activities due to weather conditions.\n"
                elif current_date['precipitation'] > 2:
                    itinerary += "\n**☔ Rain Alert:** Carry an umbrella and have indoor backup plans.\n"
            
            itinerary += "\n---\n\n"
        
        # Travel Tips
        itinerary += "## 🎒 Travel Tips\n\n"
        itinerary += "### 🧳 Packing Recommendations\n"
        
        # Weather-based packing
        max_temp = max([f['max_temp'] for f in weather['daily_forecasts']] + [20])
        min_temp = min([f['min_temp'] for f in weather['daily_forecasts']] + [15])
        
        if max_temp > 25:
            itinerary += "- Light, breathable clothing for warm weather\n"
            itinerary += "- Sun protection (hat, sunglasses, sunscreen)\n"
        if min_temp < 15:
            itinerary += "- Warm layers for cooler temperatures\n"
            itinerary += "- Light jacket or sweater\n"
        if any(f['precipitation'] > 1 for f in weather['daily_forecasts']):
            itinerary += "- Waterproof jacket or umbrella\n"
            itinerary += "- Water-resistant footwear\n"
        
        itinerary += "- Comfortable walking shoes\n"
        itinerary += "- Portable charger and camera\n"
        itinerary += "- Local currency and payment cards\n\n"
        
        # Safety and local tips
        itinerary += "### 🔒 Safety & Local Tips\n"
        itinerary += "- Keep copies of important documents\n"
        itinerary += "- Stay hydrated and take breaks\n"
        itinerary += "- Research local customs and etiquette\n"
        itinerary += "- Have emergency contacts readily available\n"
        itinerary += "- Check opening hours before visiting attractions\n\n"
        
        # Budget estimation
        itinerary += "### 💰 Estimated Budget (Per Person)\n"
        itinerary += "- Accommodation: $30-80 per night\n"
        itinerary += "- Meals: $15-30 per day\n"
        itinerary += "- Transportation: $10-25 per day\n"
        itinerary += "- Activities: $20-50 per day\n"
        itinerary += f"- **Total Estimated:** $75-185 per day for {num_days} days\n\n"
        
        itinerary += "---\n\n"
        itinerary += "*Generated by AI Trip Planning Agent*\n"
        itinerary += f"*Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        logger.info("✅ Itinerary generation completed")
        return itinerary

    def handle_error(self, exception: Exception) -> str:
        """
        Generate a fallback itinerary when errors occur
        
        Args:
            exception: The exception that occurred
            
        Returns:
            Fallback itinerary string
        """
        logger.error(f"❌ Generating fallback itinerary due to error: {str(exception)}")
        
        fallback = "# 🚨 Trip Planning - Limited Information Available\n\n"
        fallback += f"**Error encountered:** {str(exception)}\n\n"
        fallback += "Unfortunately, we couldn't retrieve all the detailed information for your trip due to API limitations. "
        fallback += "Here's a general trip planning guide:\n\n"
        
        fallback += "## 📋 General Trip Planning Checklist\n\n"
        fallback += "### Before You Go\n"
        fallback += "- [ ] Research your destination's weather forecast\n"
        fallback += "- [ ] Book accommodation in advance\n"
        fallback += "- [ ] Check visa/passport requirements\n"
        fallback += "- [ ] Notify your bank of travel plans\n"
        fallback += "- [ ] Get travel insurance\n"
        fallback += "- [ ] Pack appropriate clothing for the climate\n\n"
        
        fallback += "### During Your Trip\n"
        fallback += "- [ ] Visit popular tourist attractions\n"
        fallback += "- [ ] Try local cuisine and restaurants\n"
        fallback += "- [ ] Explore museums and cultural sites\n"
        fallback += "- [ ] Take walking tours or guided tours\n"
        fallback += "- [ ] Shop for local souvenirs\n"
        fallback += "- [ ] Take photos and create memories\n\n"
        
        fallback += "### Safety Tips\n"
        fallback += "- Keep important documents safe\n"
        fallback += "- Stay aware of your surroundings\n"
        fallback += "- Have emergency contacts available\n"
        fallback += "- Research local emergency numbers\n"
        fallback += "- Keep some cash for emergencies\n\n"
        
        fallback += "**Recommendation:** Try planning your trip again later, or consult local travel guides "
        fallback += "and tourism websites for detailed destination information.\n\n"
        
        fallback += "---\n"
        fallback += f"*Error logged on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        return fallback

    def main(self, query: str, current_date: str = None) -> str:
        """
        Main orchestration function for trip planning with enhanced logging
        
        Args:
            query: User's trip planning query
            current_date: Current date (defaults to today)
            
        Returns:
            Complete itinerary as a formatted string
        """
        if current_date is None:
            current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Enhanced trigger logging
        logger.info("🌟 === VACATION AGENT TRIGGERED ===")
        logger.info(f"🎯 Vacation agent triggered for query: {query}")
        logger.info(f"📅 Current Date: {current_date}")
        logger.info("🔄 Initializing trip planning process...")
        
        try:
            # Step 1: Parse the query
            logger.info("📝 Step 1: Parsing user query...")
            parsed_data = self.parse_query(query, current_date)
            logger.info(f"✅ Query parsed successfully - Location: {parsed_data['location']}, Days: {parsed_data['num_days']}")
            
            # Step 2: Geocode the location
            logger.info("🌍 Step 2: Geocoding location...")
            lat, lon = self.geocode_location(parsed_data['location'], self.geoapify_api_key)
            parsed_data['lat'] = lat
            parsed_data['lon'] = lon
            logger.info(f"✅ Geocoding successful - Coordinates: ({lat}, {lon})")
            
            # Step 3: Get weather forecast
            logger.info("🌤️ Step 3: Fetching weather forecast...")
            weather = self.get_weather(lat, lon, parsed_data['start_date'], parsed_data['end_date'])
            logger.info(f"✅ Weather data retrieved - Conditions: {weather.get('overall_conditions', 'unknown')}")
            
            # Step 4: Get places of interest
            logger.info("🎯 Step 4: Searching for places of interest...")
            places = self.get_places(lat, lon, weather, parsed_data['num_days'], self.geoapify_api_key, parsed_data['location'])
            logger.info(f"✅ Places search completed - Found {len(places)} places")
            
            # Step 5: Calculate routes (enhanced for multiple routes)
            logger.info("🛣️ Step 5: Calculating routes between locations...")
            routes = []
            route_count = 0
            if len(places) >= 2:
                try:
                    # Calculate routes for first 3 places for better route planning
                    max_routes = min(3, len(places) - 1)
                    for i in range(max_routes):
                        route = self.get_route(
                            places[i]['lat'], places[i]['lon'],
                            places[i + 1]['lat'], places[i + 1]['lon'],
                            self.ors_api_key
                        )
                        route['from_place'] = places[i]['name']
                        route['to_place'] = places[i + 1]['name']
                        routes.append(route)
                        route_count += 1
                        logger.info(f"🚗 Route {i+1}: {places[i]['name']} → {places[i+1]['name']} ({route['distance_km']}km)")
                except Exception as e:
                    logger.warning(f"⚠️ Route calculation failed: {str(e)}")
                    # Continue without routes
            
            logger.info(f"✅ Route calculation completed - {route_count} routes calculated")
            
            # Step 6: Generate the itinerary
            logger.info("📋 Step 6: Generating comprehensive itinerary...")
            itinerary = self.generate_itinerary(parsed_data, weather, places, routes)
            logger.info("✅ Itinerary generation completed successfully")
            
            logger.info("🎉 === VACATION AGENT COMPLETED SUCCESSFULLY ===")
            logger.info(f"📊 Summary: {parsed_data['num_days']}-day trip to {parsed_data['location']} with {len(places)} places and {len(routes)} routes")
            
            return itinerary
            
        except Exception as e:
            logger.error(f"💥 Fatal error in vacation agent: {str(e)}")
            logger.error("🔄 Switching to fallback itinerary generation...")
            return self.handle_error(e)


def main():
    """Main entry point when script is run directly with command-line argument support"""
    try:
        # Initialize the trip planning agent
        agent = TripPlanningAgent()
        
        # Check for command-line arguments
        if len(sys.argv) > 1:
            # Use command-line argument as query
            user_query = " ".join(sys.argv[1:])
            print(f"🌟 Welcome to the AI Trip Planning Agent! 🌟\n")
            print(f"Processing your query: '{user_query}'\n")
        else:
            # Use sample query for demonstration
            user_query = "Plan a 4-day trip to Lisbon next month."
            print("🌟 Welcome to the AI Trip Planning Agent! 🌟\n")
            print("No query provided via command line. Using sample query for demonstration.\n")
            print(f"Sample query: '{user_query}'\n")
            print("Usage: python trip_planner.py 'Plan a 3-day trip to Paris next week'\n")
        
        print("=" * 60)
        print("🔄 Processing your trip planning request...")
        print("=" * 60)
        
        # Generate the itinerary
        result = agent.main(user_query)
        
        # Print the result
        print("\n" + "=" * 60)
        print("📋 YOUR PERSONALIZED ITINERARY")
        print("=" * 60)
        print(result)
        
        # Additional information
        print("\n" + "=" * 60)
        print("🎯 Trip planning completed! Save this itinerary for your reference.")
        print("💡 Tip: You can also run this script with your own query:")
        print("   python trip_planner.py 'Plan a 5-day trip to Tokyo next month'")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n❌ Trip planning cancelled by user.")
        logger.info("🛑 Trip planning cancelled by user interrupt")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n💥 Critical error: {str(e)}")
        print("Please check your environment variables and try again.")
        logger.error(f"💥 Critical error in main execution: {str(e)}")
        sys.exit(1)
        sys.exit(1)


if __name__ == "__main__":
    main()
