import easyocr
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)

class InvoiceService:
    def __init__(self):
        """Initialize the InvoiceService with EasyOCR reader"""
        try:
            # Initialize EasyOCR reader for English
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            logger.info("📄 InvoiceService initialized with EasyOCR")
        except Exception as e:
            logger.error(f"❌ Failed to initialize EasyOCR: {e}")
            self.ocr_reader = None

    def classify_image_as_invoice(self, image_data: str) -> bool:
        """
        Classify if the image is an invoice using OCR text analysis
        Args:
            image_data: Base64 encoded image data
        Returns:
            bool: True if image is classified as an invoice
        """
        try:
            # Convert base64 to PIL Image
            image = self._base64_to_image(image_data)
            if not image:
                return False

            # Extract text using OCR
            ocr_text = self._extract_text_from_image(image)
            if not ocr_text:
                return False

            # Look for invoice-related keywords
            invoice_keywords = [
                'invoice', 'bill', 'receipt', 'total', 'amount', 'tax',
                'subtotal', 'payment', 'due', 'vendor', 'customer',
                'date', 'qty', 'quantity', 'price', 'item', 'description'
            ]
            
            text_lower = ocr_text.lower()
            keyword_matches = sum(1 for keyword in invoice_keywords if keyword in text_lower)
            
            # If we find at least 3 invoice-related keywords, classify as invoice
            is_invoice = keyword_matches >= 3
            
            logger.info(f"📊 Image classification: {'Invoice' if is_invoice else 'Not Invoice'} (keywords found: {keyword_matches})")
            return is_invoice
            
        except Exception as e:
            logger.error(f"❌ Error classifying image: {e}")
            return False

    def extract_invoice_details(self, image_data: str) -> Dict[str, Any]:
        """
        Extract invoice details using OCR
        Args:
            image_data: Base64 encoded image data
        Returns:
            Dict containing extracted invoice details
        """
        try:
            # Convert base64 to PIL Image
            image = self._base64_to_image(image_data)
            if not image:
                raise Exception("Failed to process image data")

            # Extract text using OCR
            ocr_text = self._extract_text_from_image(image)
            if not ocr_text:
                raise Exception("No text could be extracted from image")

            logger.info(f"📝 Extracted OCR text (first 200 chars): {ocr_text[:200]}...")

            # Extract specific invoice fields
            extracted_data = {
                'invoice_number': self._extract_invoice_number(ocr_text),
                'invoice_date': self._extract_invoice_date(ocr_text),
                'vendor_name': self._extract_vendor_name(ocr_text),
                'buyer_details': self._extract_buyer_details(ocr_text),
                'line_items': self._extract_line_items(ocr_text),
                'total_amount': self._extract_total_amount(ocr_text),
                'tax_amount': self._extract_tax_amount(ocr_text),
                'payment_terms': self._extract_payment_terms(ocr_text),
                'po_number': self._extract_po_number(ocr_text),
                'additional_details': self._extract_additional_details(ocr_text),
                'raw_ocr_text': ocr_text
            }

            logger.info(f"✅ Invoice details extracted successfully")
            return extracted_data

        except Exception as e:
            logger.error(f"❌ Error extracting invoice details: {e}")
            raise

    def validate_invoice_details(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted invoice details against comprehensive predefined rules with overall scoring
        Args:
            extracted_data: Dictionary containing extracted invoice details
        Returns:
            Dict containing validation results, overall score, and extracted information
        """
        validation_results = {}
        
        # Rule 1: Invoice Number (MANDATORY - Higher weight)
        invoice_number = extracted_data.get('invoice_number')
        if invoice_number and invoice_number.strip() and invoice_number.lower() not in ['not found', 'none', 'n/a']:
            validation_results['invoice_number'] = {
                'status': 'pass',
                'score': 15,  # Critical field
                'explanation': f"Invoice number found: {invoice_number}"
            }
        else:
            validation_results['invoice_number'] = {
                'status': 'fail',
                'score': 0,
                'explanation': "MANDATORY: Invoice number must be present, unique, and clearly visible"
            }

        # Rule 2: Invoice Date Present and Correctly Formatted
        invoice_date = extracted_data.get('invoice_date')
        if invoice_date and invoice_date.strip() and invoice_date.lower() not in ['not found', 'none', 'n/a']:
            validation_results['invoice_date'] = {
                'status': 'pass',
                'score': 10,
                'explanation': f"Invoice date found: {invoice_date}"
            }
        else:
            validation_results['invoice_date'] = {
                'status': 'fail',
                'score': 0,
                'explanation': "Invoice date should be present and correctly formatted (DD/MM/YYYY or MM-DD-YYYY)"
            }

        # Rule 3: Vendor/Store Name and Address
        vendor_name = extracted_data.get('vendor_name')
        if vendor_name and vendor_name.strip() and vendor_name.lower() not in ['not found', 'none', 'n/a']:
            validation_results['vendor_info'] = {
                'status': 'pass',
                'score': 10,
                'explanation': f"Vendor/store information found: {vendor_name}"
            }
        else:
            validation_results['vendor_info'] = {
                'status': 'fail',
                'score': 0,
                'explanation': "Vendor/store name and address must be clearly visible and accurate"
            }

        # Rule 4: Buyer/Client Name and Address
        buyer_details = extracted_data.get('buyer_details')
        if buyer_details and buyer_details.strip() and buyer_details.lower() not in ['not found', 'none', 'n/a']:
            validation_results['buyer_info'] = {
                'status': 'pass',
                'score': 8,
                'explanation': f"Buyer/client information found: {buyer_details}"
            }
        else:
            validation_results['buyer_info'] = {
                'status': 'fail',
                'score': 0,
                'explanation': "Buyer/client name and address should be present where applicable"
            }

        # Rule 5: List of Items/Services
        line_items = extracted_data.get('line_items', [])
        if line_items and len(line_items) > 0:
            validation_results['items_services'] = {
                'status': 'pass',
                'score': 12,
                'explanation': f"{len(line_items)} item(s)/service(s) with description, quantity, unit price detected"
            }
        else:
            validation_results['items_services'] = {
                'status': 'fail',
                'score': 0,
                'explanation': "List of items/services with description, quantity, and unit price is required"
            }

        # Rule 6: Total Amount
        total_amount = extracted_data.get('total_amount')
        if total_amount and total_amount.strip() and total_amount.lower() not in ['not found', 'none', 'n/a']:
            validation_results['total_amount'] = {
                'status': 'pass',
                'score': 12,
                'explanation': f"Total amount clearly stated: {total_amount}"
            }
        else:
            validation_results['total_amount'] = {
                'status': 'fail',
                'score': 0,
                'explanation': "Total amount must be clearly stated and numeric"
            }

        # Rule 7: Taxes (VAT, GST, Sales Tax)
        tax_amount = extracted_data.get('tax_amount')
        if tax_amount and tax_amount.strip() and tax_amount.lower() not in ['not found', 'none', 'n/a']:
            validation_results['taxes'] = {
                'status': 'pass',
                'score': 10,
                'explanation': f"Tax information found: {tax_amount}"
            }
        else:
            validation_results['taxes'] = {
                'status': 'fail',
                'score': 0,
                'explanation': "Taxes (VAT, GST, Sales Tax) should be displayed with correct amounts and rates"
            }

        # Rule 8: Payment Terms and Due Dates
        payment_terms = extracted_data.get('payment_terms')
        if payment_terms and payment_terms.strip() and payment_terms.lower() not in ['not found', 'none', 'n/a']:
            validation_results['payment_terms'] = {
                'status': 'pass',
                'score': 8,
                'explanation': f"Payment terms found: {payment_terms}"
            }
        else:
            validation_results['payment_terms'] = {
                'status': 'fail',
                'score': 0,
                'explanation': "Payment terms and due dates should be included if available"
            }

        # Rule 9: Purchase Order Number
        po_number = extracted_data.get('po_number')
        if po_number and po_number.strip() and po_number.lower() not in ['not found', 'none', 'n/a']:
            validation_results['po_number'] = {
                'status': 'pass',
                'score': 5,
                'explanation': f"Purchase order number found: {po_number}"
            }
        else:
            validation_results['po_number'] = {
                'status': 'fail',
                'score': 0,
                'explanation': "Purchase order number should be present if applicable for matching"
            }

        # Rule 10: Additional Details and Image Quality
        raw_text = extracted_data.get('raw_ocr_text', '')
        additional_details = extracted_data.get('additional_details', '')
        if len(raw_text) > 50 or additional_details:  # Sufficient text extracted
            validation_results['additional_details'] = {
                'status': 'pass',
                'score': 10,
                'explanation': "Additional details (discounts, shipping, contact info) and image quality sufficient"
            }
        else:
            validation_results['additional_details'] = {
                'status': 'fail',
                'score': 0,
                'explanation': "Additional details (discounts, shipping costs, contact information) and image quality insufficient"
            }

        # Calculate overall score
        total_score = sum(rule['score'] for rule in validation_results.values())
        max_possible_score = 100  # 15+10+10+8+12+12+10+8+5+10 = 100
        score_percentage = (total_score / max_possible_score) * 100
        
        # Determine overall status
        if score_percentage >= 85:
            overall_status = "EXCELLENT"
            status_description = "Invoice meets all validation requirements with high quality"
        elif score_percentage >= 70:
            overall_status = "GOOD"
            status_description = "Invoice meets most validation requirements"
        elif score_percentage >= 50:
            overall_status = "ACCEPTABLE" 
            status_description = "Invoice has some missing information but can be processed"
        else:
            overall_status = "POOR"
            status_description = "Invoice has significant issues and needs manual review"
        
        # Special case: If invoice number (mandatory) is missing, mark as poor
        if validation_results['invoice_number']['status'] == 'fail':
            overall_status = "POOR"
            status_description = "Invoice number is mandatory - cannot process without it"
            
        passed_rules = sum(1 for r in validation_results.values() if r['status'] == 'pass')
        total_rules = len(validation_results)
        
        logger.info(f"📋 Invoice validation: {total_score}/100 points | {passed_rules}/{total_rules} rules passed | Status: {overall_status}")
        
        return {
            'validation_results': validation_results,
            'overall_score': total_score,
            'score_percentage': round(score_percentage, 1),
            'overall_status': overall_status,
            'status_description': status_description,
            'rules_passed': passed_rules,
            'total_rules': total_rules,
            'extracted_data': extracted_data
        }

    def format_invoice_report(self, validation_results: Dict[str, Any]) -> str:
        """
        Format a structured invoice analysis report with hierarchical structure followed by structured table format:
        1. Overall Score (at top)
        2. Extracted Information in structured table format (like the user's example)
        3. Validation Rules (with status for each field)
        Args:
            validation_results: Complete validation results including scores and extracted data
        Returns:
            Formatted report string
        """
        extracted_data = validation_results['extracted_data']
        rules = validation_results['validation_results']
        
        report = "## 📄 Invoice Analysis Report\n\n"
        
        # 1. OVERALL SCORE (at the top)
        report += f"### 🏆 Overall Assessment\n\n"
        report += f"**Score:** {validation_results['overall_score']}/100 ({validation_results['score_percentage']}%)\n\n"
        report += f"**Status:** {validation_results['overall_status']}\n\n"
        report += f"**Description:** {validation_results['status_description']}\n\n"
        report += f"**Rules Passed:** {validation_results['rules_passed']}/{validation_results['total_rules']}\n\n"
        
        # 2. EXTRACTED INFORMATION in structured table format
        report += "### 📊 Extracted Information\n\n"
        
        # Get key fields for structured format
        invoice_number = extracted_data.get('invoice_number', 'Not Found')
        invoice_date = extracted_data.get('invoice_date', 'Not Found')
        due_date = extracted_data.get('payment_terms', 'Not Found')  # Use payment terms as due date
        vendor_name = extracted_data.get('vendor_name', 'Not Found')
        buyer_details = extracted_data.get('buyer_details', 'Not Found')
        total_amount = extracted_data.get('total_amount', 'Not Found')
        
        # Invoice Details Table
        report += "### **Invoice Details**\n\n"
        report += "| Invoice No. | Invoice Date | Due Date |\n"
        report += "| :--- | :--- | :--- |\n"
        report += f"| {invoice_number} | {invoice_date} | {due_date} |\n\n"
        
        report += "---\n\n"
        
        # Parties Table
        report += "### **Parties**\n\n"
        report += "| From | To |\n"
        report += "| :--- | :--- |\n"
        
        # Split vendor and buyer details for better presentation
        vendor_lines = vendor_name.split('\n') if vendor_name != 'Not Found' else ['Not Found']
        buyer_lines = buyer_details.split('\n') if buyer_details != 'Not Found' else ['Not Found']
        
        # Get the maximum number of lines to display both parties
        max_lines = max(len(vendor_lines), len(buyer_lines))
        
        for i in range(max_lines):
            vendor_line = vendor_lines[i] if i < len(vendor_lines) else ''
            buyer_line = buyer_lines[i] if i < len(buyer_lines) else ''
            
            # Bold the first line (usually the main name)
            if i == 0:
                vendor_line = f"**{vendor_line}**" if vendor_line else ''
                buyer_line = f"**{buyer_line}**" if buyer_line else ''
            
            report += f"| {vendor_line} | {buyer_line} |\n"
        
        report += "\n---\n\n"
        
        # Services/Items Table
        report += "### **Services Rendered**\n\n"
        line_items = extracted_data.get('line_items', [])
        
        if line_items and len(line_items) > 0:
            report += "| Item | Description | HRS/QTY | Rate | Subtotal |\n"
            report += "| :--- | :--- | :--- | :--- | :--- |\n"
            
            for i, item in enumerate(line_items[:5], 1):  # Show first 5 items
                # Try to parse item details - this is basic parsing, might need enhancement
                item_parts = item.split()
                if len(item_parts) >= 3:
                    description = ' '.join(item_parts[:-2]) if len(item_parts) > 3 else item_parts[0]
                    qty = item_parts[-2] if len(item_parts) > 2 else '1'
                    rate = item_parts[-1] if len(item_parts) > 1 else 'N/A'
                    subtotal = rate  # Simple assumption
                else:
                    description = item
                    qty = '1'
                    rate = 'N/A'
                    subtotal = 'N/A'
                
                # Bold the description
                description = f"**{description}**"
                report += f"| {description} | {description} | {qty} | {rate} | {subtotal} |\n"
            
            if len(line_items) > 5:
                report += f"| ... | ... | ... | ... | ... |\n"
                report += f"| *({len(line_items) - 5} more items)* | | | | |\n"
        else:
            report += "| Item | Description | HRS/QTY | Rate | Subtotal |\n"
            report += "| :--- | :--- | :--- | :--- | :--- |\n"
            report += "| No items detected | | | | |\n"
        
        report += "\n---\n\n"
        
        # Invoice Summary Table
        report += "### **Invoice Summary**\n\n"
        report += "| | |\n"
        report += "| :--- | ---: |\n"
        
        # Extract subtotal if available, otherwise use total
        tax_amount = extracted_data.get('tax_amount', '0.00')
        if total_amount != 'Not Found':
            report += f"| **Subtotal** | {total_amount} |\n"
            if tax_amount != 'Not Found' and tax_amount != '0.00':
                report += f"| **Tax** | {tax_amount} |\n"
            report += f"| **Total** | **{total_amount}** |\n"
        else:
            report += "| **Subtotal** | Not Found |\n"
            report += "| **Total** | **Not Found** |\n"
        
        report += "\n---\n\n"
        
        # 3. VALIDATION RULES (with status for each field)
        report += "### ✅ Field Validation Results\n\n"
        
        rule_details = [
            ('invoice_number', '🔢 Invoice Number', 'MANDATORY - Must be present, unique, and clearly visible'),
            ('invoice_date', '📅 Invoice Date', 'Should be present and correctly formatted (DD/MM/YYYY or MM-DD-YYYY)'),
            ('vendor_info', '🏢 Vendor/Store Name and Address', 'Clearly visible and accurate'),
            ('buyer_info', '👤 Buyer/Client Name and Address', 'Present where applicable'),
            ('items_services', '📦 List of Items/Services', 'Description of products/services, quantity, unit price'),
            ('total_amount', '💰 Total Amount', 'Clearly stated and numeric'),
            ('taxes', '📊 Taxes (VAT, GST, Sales Tax)', 'Displayed with correct amounts and tax rates'),
            ('payment_terms', '📋 Payment Terms and Due Dates', 'If available'),
            ('po_number', '📝 Purchase Order Number', 'If applicable, for matching'),
            ('additional_details', '➕ Additional Details', 'Discounts, shipping costs, contact information')
        ]
        
        for rule_key, rule_label, rule_description in rule_details:
            rule = rules[rule_key]
            status_emoji = "✅" if rule['status'] == 'pass' else "❌"
            score = rule['score']
            
            # Determine max score for this rule
            max_scores = {
                'invoice_number': 15, 'invoice_date': 10, 'vendor_info': 10, 'buyer_info': 8,
                'items_services': 12, 'total_amount': 12, 'taxes': 10, 'payment_terms': 8,
                'po_number': 5, 'additional_details': 10
            }
            max_score = max_scores.get(rule_key, 10)
            
            report += f"**{rule_label}:** {status_emoji} {rule['status'].upper()} - {score}/{max_score} points\n"
            report += f"   └─ **Rule:** {rule_description}\n"
            report += f"   └─ **Result:** {rule['explanation']}\n\n"
        
        # Additional Status Information
        if validation_results['overall_status'] == 'POOR' and rules['invoice_number']['status'] == 'fail':
            report += "### ⚠️ Critical Issue\n\n"
            report += "**Invoice Number is mandatory** and must be present for processing. This is a critical field that cannot be missing.\n\n"
        
        # Storage Information
        if validation_results['score_percentage'] >= 50:
            report += "### 📁 Processing Status\n\n"
            report += "✅ **Invoice data has been extracted and stored** in the database for future reference and search.\n\n"
        else:
            report += "### ⚠️ Processing Note\n\n"
            report += "📝 **Invoice requires manual review** - Multiple critical fields are missing or unclear. Please verify the image quality and completeness.\n\n"
        
        return report

    def _base64_to_image(self, image_data: str) -> Optional[Image.Image]:
        """Convert base64 string to PIL Image"""
        try:
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
        except Exception as e:
            logger.error(f"❌ Error converting base64 to image: {e}")
            return None

    def _extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from image using EasyOCR"""
        try:
            if not self.ocr_reader:
                raise Exception("OCR reader not initialized")
            
            # Convert PIL Image to numpy array for EasyOCR
            image_array = np.array(image)
            
            # Use EasyOCR to extract text
            results = self.ocr_reader.readtext(image_array)
            
            # Combine all detected text
            extracted_text = ' '.join([result[1] for result in results if result[2] > 0.3])  # Confidence threshold
            
            return extracted_text.strip()
        except Exception as e:
            logger.error(f"❌ Error extracting text with OCR: {e}")
            return ""

    def _extract_invoice_number(self, text: str) -> Optional[str]:
        """Extract invoice number from OCR text"""
        patterns = [
            r'invoice\s*#?\s*:?\s*([A-Z0-9\-]+)',
            r'inv\s*#?\s*:?\s*([A-Z0-9\-]+)',
            r'#\s*([A-Z0-9\-]{3,})',
            r'invoice\s+number\s*:?\s*([A-Z0-9\-]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None

    def _extract_invoice_date(self, text: str) -> Optional[str]:
        """Extract invoice date from OCR text"""
        date_patterns = [
            r'date\s*:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
            r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
            r'(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})',
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0] if isinstance(matches[0], str) else matches[0][0]
        
        return None

    def _extract_vendor_name(self, text: str) -> Optional[str]:
        """Extract vendor/store name from OCR text"""
        # Look for text at the beginning of the document (usually vendor name)
        lines = text.split('\n')
        
        # Skip empty lines and find first substantial line
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if len(line) > 3 and not re.match(r'^\d+[\.\-\/]', line):  # Not just numbers or dates
                # Clean up common OCR artifacts
                cleaned_line = re.sub(r'[^\w\s&\-\.]', '', line)
                if len(cleaned_line) > 3:
                    return cleaned_line[:50]  # Limit length
        
        return None

    def _extract_line_items(self, text: str) -> List[str]:
        """Extract line items from OCR text"""
        lines = text.split('\n')
        line_items = []
        
        for line in lines:
            line = line.strip()
            # Look for lines that might be items (contain both text and numbers/prices)
            if (len(line) > 5 and 
                re.search(r'\d+\.?\d*', line) and  # Contains numbers
                re.search(r'[a-zA-Z]', line) and   # Contains letters
                not re.match(r'^(total|subtotal|tax|amount|date|invoice)', line, re.IGNORECASE)):
                line_items.append(line[:100])  # Limit item length
        
        return line_items[:10]  # Return first 10 items to avoid clutter

    def _extract_total_amount(self, text: str) -> Optional[str]:
        """Extract total amount from OCR text"""
        patterns = [
            r'total\s*:?\s*\$?(\d+\.?\d*)',
            r'amount\s*due\s*:?\s*\$?(\d+\.?\d*)',
            r'grand\s*total\s*:?\s*\$?(\d+\.?\d*)',
            r'final\s*amount\s*:?\s*\$?(\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount = match.group(1)
                return f"${amount}" if not text[match.start():match.end()].startswith('$') else f"${amount}"
        
        return None

    def _extract_tax_amount(self, text: str) -> Optional[str]:
        """Extract tax amount from OCR text"""
        patterns = [
            r'tax\s*:?\s*\$?(\d+\.?\d*)',
            r'vat\s*:?\s*\$?(\d+\.?\d*)',
            r'sales\s*tax\s*:?\s*\$?(\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount = match.group(1)
                return f"${amount}"
        
        return None

    def _extract_buyer_details(self, text: str) -> Optional[str]:
        """Extract buyer details from OCR text"""
        # Look for patterns that might indicate buyer information
        buyer_patterns = [
            r'bill\s*to\s*:?\s*([^\n]+)',
            r'customer\s*:?\s*([^\n]+)',
            r'buyer\s*:?\s*([^\n]+)',
            r'client\s*:?\s*([^\n]+)',
            r'sold\s*to\s*:?\s*([^\n]+)'
        ]
        
        for pattern in buyer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()[:100]  # Limit length
        
        return None

    def _extract_payment_terms(self, text: str) -> Optional[str]:
        """Extract payment terms and due dates from OCR text"""
        payment_patterns = [
            r'payment\s*terms?\s*:?\s*([^\n]+)',
            r'due\s*date\s*:?\s*([^\n]+)',
            r'terms?\s*:?\s*([^\n]+)',
            r'net\s*(\d+)\s*days?',
            r'due\s*in\s*(\d+)\s*days?',
            r'payment\s*due\s*:?\s*([^\n]+)'
        ]
        
        for pattern in payment_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()[:100] if len(match.groups()) > 0 else match.group(0).strip()[:100]
        
        return None

    def _extract_po_number(self, text: str) -> Optional[str]:
        """Extract purchase order number from OCR text"""
        po_patterns = [
            r'p\.?o\.?\s*#?\s*:?\s*([A-Z0-9\-]+)',
            r'purchase\s*order\s*#?\s*:?\s*([A-Z0-9\-]+)',
            r'order\s*#?\s*:?\s*([A-Z0-9\-]+)',
            r'po\s*number\s*:?\s*([A-Z0-9\-]+)'
        ]
        
        for pattern in po_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None

    def _extract_additional_details(self, text: str) -> Optional[str]:
        """Extract additional details like discounts, shipping, contact info"""
        additional_info = []
        
        # Look for discounts
        discount_patterns = [
            r'discount\s*:?\s*([^\n]+)',
            r'savings?\s*:?\s*([^\n]+)',
            r'(\d+)%\s*off'
        ]
        
        for pattern in discount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                additional_info.append(f"Discount: {match}")
        
        # Look for shipping
        shipping_patterns = [
            r'shipping\s*:?\s*([^\n]+)',
            r'delivery\s*:?\s*([^\n]+)',
            r'freight\s*:?\s*([^\n]+)'
        ]
        
        for pattern in shipping_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                additional_info.append(f"Shipping: {match}")
        
        # Look for contact information
        contact_patterns = [
            r'phone\s*:?\s*([^\n]+)',
            r'email\s*:?\s*([^\n]+)',
            r'contact\s*:?\s*([^\n]+)',
            r'tel\s*:?\s*([^\n]+)'
        ]
        
        for pattern in contact_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                additional_info.append(f"Contact: {match}")
        
        return "; ".join(additional_info[:3]) if additional_info else None  # Limit to first 3 items
