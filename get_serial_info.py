#!/usr/bin/env python3
"""
Script to get serial information from /dev/ttyACM0
"""

import serial
import serial.tools.list_ports
import sys
import time

def get_serial_info(port_name='/dev/ttyACM0'):
    """Get detailed information about a serial port"""
    
    print(f"Getting information for serial port: {port_name}")
    print("=" * 50)
    
    # Check if port exists in list of available ports
    available_ports = serial.tools.list_ports.comports()
    print("Available serial ports:")
    for port in available_ports:
        print(f"  - {port.device}: {port.description}")
        if port.device == port_name:
            print(f"    Manufacturer: {port.manufacturer}")
            print(f"    Product: {port.product}")
            print(f"    Serial Number: {port.serial_number}")
            print(f"    VID: {port.vid}")
            print(f"    PID: {port.pid}")
    print()
    
    try:
        # Try to open the serial port
        ser = serial.Serial()
        ser.port = port_name
        ser.baudrate = 115200  # Common baudrate, adjust as needed
        ser.timeout = 1
        
        print(f"Attempting to open {port_name}...")
        ser.open()
        
        if ser.is_open:
            print(f"✓ Successfully opened {port_name}")
            
            # Get port information
            print("\nPort Configuration:")
            print(f"  Port: {ser.port}")
            print(f"  Baudrate: {ser.baudrate}")
            print(f"  Bytesize: {ser.bytesize}")
            print(f"  Parity: {ser.parity}")
            print(f"  Stopbits: {ser.stopbits}")
            print(f"  Timeout: {ser.timeout}")
            print(f"  Write Timeout: {ser.write_timeout}")
            print(f"  XON/XOFF: {ser.xonxoff}")
            print(f"  RTS/CTS: {ser.rtscts}")
            print(f"  DSR/DTR: {ser.dsrdtr}")
            
            # Check control lines
            print("\nControl Lines:")
            try:
                print(f"  CTS (Clear To Send): {ser.cts}")
                print(f"  DSR (Data Set Ready): {ser.dsr}")
                print(f"  RI (Ring Indicator): {ser.ri}")
                print(f"  CD (Carrier Detect): {ser.cd}")
            except Exception as e:
                print(f"  Could not read control lines: {e}")
            
            # Try to read some data (with timeout)
            print("\nTrying to read data (timeout: 2 seconds)...")
            ser.timeout = 2
            try:
                data = ser.read(100)  # Try to read up to 100 bytes
                if data:
                    print(f"  Received {len(data)} bytes:")
                    print(f"  Raw: {data}")
                    try:
                        decoded = data.decode('utf-8', errors='replace')
                        print(f"  Decoded: {repr(decoded)}")
                    except:
                        print(f"  Could not decode as UTF-8")
                else:
                    print("  No data received within timeout period")
            except Exception as e:
                print(f"  Error reading data: {e}")
            
            # Close the port
            ser.close()
            print(f"\n✓ Closed {port_name}")
            
        else:
            print(f"✗ Failed to open {port_name}")
            
    except serial.SerialException as e:
        print(f"✗ Serial exception: {e}")
    except PermissionError as e:
        print(f"✗ Permission error: {e}")
        print("  Try running with sudo or add your user to the dialout group:")
        print("  sudo usermod -a -G dialout $USER")
        print("  (then logout and login again)")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

def list_all_serial_ports():
    """List all available serial ports"""
    print("All available serial ports:")
    print("=" * 30)
    
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found")
        return
    
    for i, port in enumerate(ports, 1):
        print(f"{i}. {port.device}")
        print(f"   Description: {port.description}")
        print(f"   Manufacturer: {port.manufacturer}")
        print(f"   Product: {port.product}")
        print(f"   Serial Number: {port.serial_number}")
        print(f"   VID:PID: {port.vid}:{port.pid}")
        print()

if __name__ == "__main__":
    print("Serial Port Information Tool")
    print("=" * 40)
    
    # List all ports first
    list_all_serial_ports()
    
    # Get specific info for /dev/ttyACM0
    target_port = '/dev/ttyACM0'
    if len(sys.argv) > 1:
        target_port = sys.argv[1]
    
    get_serial_info(target_port)