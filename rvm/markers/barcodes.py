import cv2
from typing import List, Optional, Set
from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol

from rvm.core.types import QRCode, BarCode

class BarCodesDetector:
    """Detected barcode and QR Code detector using pyzbar."""

    def __init__(self, symbols: Optional[Set[ZBarSymbol]] = None,
                 convert_to_grayscale: bool = True,
                 enhance_image: bool = False):
                 
        """
        Initialize the barcode detector.
        
        Args:
            symbols: Set of barcode symbols to detect. If None, detects all supported types.
            convert_to_grayscale: Whether to convert image to grayscale for better detection.
            enhance_image: Whether to apply image enhancement historgram equalization technique.
        """

        # Configure which barcode types to detect
        self.symbols = symbols or {
            ZBarSymbol.QRCODE,
            ZBarSymbol.CODE128,
            ZBarSymbol.CODE39,
            ZBarSymbol.CODE93,
            ZBarSymbol.CODABAR,
            ZBarSymbol.EAN8,
            ZBarSymbol.EAN13,
            ZBarSymbol.UPCA,
            ZBarSymbol.UPCE,
            ZBarSymbol.I25,
            ZBarSymbol.DATABAR,
            ZBarSymbol.DATABAR_EXP,
        }
        
        self.convert_to_grayscale = convert_to_grayscale
        self.enhance_image = enhance_image

    def _preprocess_image(self, image):
        """
        Preprocess the image for better detection using histogram equalization.
        
        Args:
            image (np.ndarray): Input BGR image.
            
        Returns:
            np.ndarray: Preprocessed image.
        """
        # Convert BGR to RGB for pyzbar
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.convert_to_grayscale:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
        
        if self.enhance_image:
            # Apply histogram equalization for better contrast
            if len(processed_image.shape) == 2:  # Grayscale
                processed_image = cv2.equalizeHist(processed_image)
            else:  # RGB
                processed_image = cv2.convertScaleAbs(processed_image, alpha=1.2, beta=10)
        
        return processed_image


    def detect(self, image) -> tuple[List[QRCode], List[BarCode]]:
        """
        Detect every barcode type present in the image
        
        Args:
            image (np.ndarray): Input BGR image.
            
        Returns:
            tuple: (List of QR codes, List of barcodes)
        """

        processed_image = self._preprocess_image(image)
        detected_codes = pyzbar.decode(processed_image, symbols=self.symbols)
        
        qr_codes: List[QRCode] = []
        barcodes: List[BarCode] = []
        
        for code in detected_codes:
            try:
                corners = [(int(point.x), int(point.y)) for point in code.polygon]
                data = code.data.decode('utf-8')
                
                if code.type == 'QRCODE':
                    qr_codes.append(QRCode(data=data, corners=corners))
                else:
                    barcodes.append(BarCode(data=data, corners=corners))

            except UnicodeDecodeError:
                # Handle non-UTF8 encoded data
                continue
                
        return qr_codes, barcodes
    

    def detect_qr(self, image) -> List[QRCode]:
        """
        Detect only QR codes in the image.
        
        Args:
            image (np.ndarray): Input BGR image.
            
        Returns:
            List[QRCode]: List of QR codes with data and corners.
        """
        qr_codes, _ = self.detect_all(image)
        return qr_codes


    def detect_barcodes(self, image) -> List[BarCode]:
        """
        Detect barcodes in the image (excludes QR codes).
        
        Args:
            image (np.ndarray): Input BGR image.
            
        Returns:
            List[BarCode]: List of barcodes with data and corners.
        """
        _, barcodes = self.detect_all(image)
        return barcodes
