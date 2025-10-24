#!/usr/bin/env python3
"""
Script to read velocity and yaw data from /dev/ttyACM0
"""

import serial
import sys

def read_serial_data(port_name='/dev/ttyACM0', baudrate=115200):
    """Read and parse serial data from the port"""
    
    try:
        # Open the serial port
        ser = serial.Serial(port_name, baudrate, timeout=1)
        
        print(f"Reading (vx, vy, yaw) from {port_name} at {baudrate} baud...")
        print("Press Ctrl+C to stop\n")
        
        while True:
            try:
                # Read a line from serial
                line = ser.readline()
                
                if line:
                    # Decode and strip whitespace
                    data_str = line.decode('utf-8').strip()
                    
                    if data_str:
                        # Parse the comma-separated values
                        values = data_str.split(',')
                        
                        if len(values) == 3:
                            try:
                                vx = float(values[0].strip())
                                vy = float(values[1].strip())
                                yaw = float(values[2].strip())
                                
                                # Print as vector
                                print(f"(vx={vx:.3f}, vy={vy:.3f}, yaw={yaw:.3f})")
                                
                            except ValueError:
                                # Skip lines that can't be parsed
                                pass
                                
            except KeyboardInterrupt:
                print("\nStopping...")
                break
                
        ser.close()
        
    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except PermissionError:
        print("Permission denied. Try running with sudo or add user to dialout group:")
        print("  sudo usermod -a -G dialout $USER")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    port = '/dev/ttyACM0'
    if len(sys.argv) > 1:
        port = sys.argv[1]
    
    read_serial_data(port)