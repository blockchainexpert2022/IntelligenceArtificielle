import socket
import ipaddress
from concurrent.futures import ThreadPoolExecutor

def scan_ip(ip):
    try:
        # Attempt to create a socket connection to the IP address
        socket.setdefaulttimeout(1)  # Set a timeout for the connection
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((str(ip), 135))  # Port 80 is commonly used for HTTP
        return f"{ip} is online"
    except (socket.timeout, ConnectionRefusedError):
        return ""#"f"{ip} is offline"
    except Exception as e:
        return f"Error scanning {ip}: {e}"
    finally:
        s.close()

def scan_network(network):
    # Create a list of IP addresses in the network
    ip_list = [ip for ip in ipaddress.IPv4Network(network).hosts()]
    
    # Use ThreadPoolExecutor to scan multiple IPs concurrently
    with ThreadPoolExecutor(max_workers=50) as executor:
        results = executor.map(scan_ip, ip_list)
    
    return list(results)

if __name__ == "__main__":
    # Change '192.168.1.0/24' to your local network
    network = '10.1.1.0/24'
    print("Scanning network:", network)
    scan_results = scan_network(network)
    
    for result in scan_results:
        if result != "":
            print(result)
