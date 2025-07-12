#class my_class(object):
#    pass

from scapy.all import *
import pandas as pd
import datetime
import socket

class sniffing_to_file:

    def __init__(self, num_pack_da_catturare):
        #self.interfaccia_in_uso = 
        #print("Interfaccia in uso:", interfaccia_in_uso)
        #self.INTERFACCIA = interfaccia_in_uso
        self.INTERFACCIA = self.__rileva_interfaccia_attiva()
        self.PACCHETTI_DA_CATTURARE = num_pack_da_catturare
        self.FILTRO_BPF = "ip"
        self.FILE_PARQUET = f"sniff_rich_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        self.dati = []
    
    @staticmethod
    def __rileva_interfaccia_attiva():
        try:
            # 1. Scopri l'IP locale usato per uscire (verso internet)
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))  # IP di Google DNS
            ip_locale = s.getsockname()[0]
            s.close()
            
            # === CONFIG ===
            #interfacce = get_if_list()
            #print("Interfacce disponibili:")
            #for i, iface in enumerate(interfacce):
            #    print(f"{i}: {iface}")

            # 2. Associa l'IP locale all'interfaccia
            for iface in get_if_list():
                try:
                    if get_if_addr(iface) == ip_locale:
                        return iface
                except:
                    continue
        except Exception as e:
            print("Errore nel rilevamento:", e)
            return None
    
    # === RACCOLTA ===
    
    
    def __pacchetto_to_dict(self, pkt):
        if IP in pkt:
            info = {
                "timestamp": pkt.time,
                "src_ip": pkt[IP].src,
                "dst_ip": pkt[IP].dst,
                "proto": pkt[IP].proto,
                "ttl": pkt[IP].ttl,
                "ip_proto_name": pkt[IP].name,
                "packet_length": len(pkt),
                "payload_len": len(pkt[Raw]) if Raw in pkt else 0,
                "label": "UNLABELED"  # Placeholder (deve diventare POS, BND o NEG)
            }
    
            if TCP in pkt:
                info.update({
                    "sport": pkt[TCP].sport,
                    "dport": pkt[TCP].dport,
                    "protocol": "TCP",
                    "tcp_flags": pkt[TCP].flags
                })
            elif UDP in pkt:
                info.update({
                    "sport": pkt[UDP].sport,
                    "dport": pkt[UDP].dport,
                    "protocol": "UDP",
                    "tcp_flags": None
                })
            else:
                info.update({
                    "sport": None,
                    "dport": None,
                    "protocol": "OTHER",
                    "tcp_flags": None
                })
    
            return info
        return None
    
    def __raccogli(self, pkt):
        r = self.__pacchetto_to_dict(pkt)
        if r:
            self.dati.append(r)
    
    def avvio_sniffing(self):
        print(f"Sniffing su '{self.INTERFACCIA}'... {self.PACCHETTI_DA_CATTURARE} pacchetti.")
        sniff(
            iface=self.INTERFACCIA,
            filter=self.FILTRO_BPF,
            prn=self.__raccogli,
            count=self.PACCHETTI_DA_CATTURARE,
            store=False
        )
        # === Salvataggio della collezione dei pacchetti in csv e parquet ===
        df = pd.DataFrame(self.dati)
        df = df.astype(str)
        df.to_csv('chekresult.csv')
        #può essere interessante generare dati sintetici per l'addestramento decommentare il successivo
        #df.to_parquet(self.FILE_PARQUET, index=False)
        #print(f"\nSniffing completato. File salvato in: {self.FILE_PARQUET}")
        return df

#test run class numero di pacchetti che voglio catturare 
df = sniffing_to_file(200).avvio_sniffing()
print(df)