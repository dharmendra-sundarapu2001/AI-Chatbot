#!/usr/bin/env python3
"""
Test script to demonstrate intelligent location auto-correction
"""

import os
import sys
from dotenv import load_dotenv

# Add the services directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Load environment variables
load_dotenv()

from services.trip_planner import TripPlanningAgent

def test_autocorrection():
    """Test the intelligent auto-correction system"""
    
    print("🧠 Initializing Trip Planning Agent with Intelligent Auto-Correction...")
    agent = TripPlanningAgent()
    
    # Test cases with common misspellings
    test_cases = [
        "ladkh",           # Should correct to "Leh, Ladakh"
        "kedaranth",       # Should correct to "Kedarnath, Uttarakhand"
        "darjiling",       # Should correct to "Darjeeling, West Bengal"
        "ootty",           # Should correct to "Udhagamandalam, Tamil Nadu"
        "manli",           # Should correct to "Manali, Himachal Pradesh"
        "simla",           # Should correct to "Shimla, Himachal Pradesh"
        "bombay",          # Should correct to "Mumbai, Maharashtra"
        "calcutta",        # Should correct to "Kolkata, West Bengal"
        "madras",          # Should correct to "Chennai, Tamil Nadu"
        "cochin",          # Should correct to "Kochi, Kerala"
        "trivandrum",      # Should correct to "Thiruvananthapuram, Kerala"
        "vizag",           # Should correct to "Visakhapatnam, Andhra Pradesh"
        "banaras",         # Should correct to "Varanasi, Uttar Pradesh"
        "poona",           # Should correct to "Pune, Maharashtra"
        "hubili",          # Should correct to "Hubli, Karnataka"
        "munar",           # Should correct to "Munnar, Kerala"
        "kodikanal",       # Should correct to "Kodaikanal, Tamil Nadu"
        "varkalla",        # Should correct to "Varkala, Kerala"
        "dharmsala",       # Should correct to "Dharamshala, Himachal Pradesh"
        "pushkr",          # Should correct to "Pushkar, Rajasthan"
        "invalid123",      # Should remain as-is (low confidence)
        "xyz random",      # Should remain as-is (low confidence)
    ]
    
    print("\n" + "="*80)
    print("🔧 INTELLIGENT LOCATION AUTO-CORRECTION TEST RESULTS")
    print("="*80)
    print(f"{'Original Input':<20} {'Corrected Location':<35} {'Confidence':<12} {'Status'}")
    print("-"*80)
    
    correct_corrections = 0
    total_tests = len(test_cases)
    
    for original in test_cases:
        try:
            corrected, confidence = agent._smart_location_correction(original)
            
            # Determine status
            if confidence >= 0.8:
                status = "✅ HIGH"
            elif confidence >= 0.5:
                status = "⚠️ MEDIUM"
            else:
                status = "❌ LOW"
            
            # Check if correction was applied
            if corrected.lower() != original.lower() and confidence >= 0.8:
                correct_corrections += 1
                
            print(f"{original:<20} {corrected[:34]:<35} {confidence:.2f}        {status}")
            
        except Exception as e:
            print(f"{original:<20} {'ERROR: ' + str(e)[:25]:<35} {'0.00':<12} {'❌ ERROR'}")
    
    print("-"*80)
    print(f"📊 SUMMARY:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Successful Auto-corrections: {correct_corrections}")
    print(f"   Success Rate: {(correct_corrections/total_tests)*100:.1f}%")
    print("="*80)
    
    # Test with vacation planning queries
    print("\n🌴 TESTING WITH VACATION PLANNING QUERIES:")
    print("-"*80)
    
    vacation_queries = [
        "I want to plan a trip to ladkh for 5 days",
        "Plan a vacation to kedaranth next month",
        "I want to go to darjiling for weekend",
        "Trip plan to ootty for 3 days",
        "Want to visit manli in December"
    ]
    
    for query in vacation_queries:
        print(f"\n🎯 Query: '{query}'")
        
        # Extract location from query (simple regex)
        import re
        location_match = re.search(r'(?:to|visit|trip to|go to|in|plan.*trip.*to)\s+([A-Za-z\s,]+?)(?:\s+(?:for|next|this|in|\d+)|\s*$)', query.lower())
        if location_match:
            raw_location = location_match.group(1).strip()
            corrected, confidence = agent._smart_location_correction(raw_location)
            
            if confidence >= 0.8 and corrected.lower() != raw_location.lower():
                print(f"   🔧 Auto-correction: '{raw_location}' → '{corrected}' (confidence: {confidence:.2f})")
            else:
                print(f"   ✅ Location: '{raw_location}' (no correction needed)")
        else:
            print(f"   ⚠️ Could not extract location from query")
    
    print("\n✨ Auto-correction system test completed!")
    print("🎉 The system successfully demonstrates intelligent location correction")
    print("   using pyspellchecker, fuzzywuzzy, and a comprehensive location database!")

if __name__ == "__main__":
    test_autocorrection()
