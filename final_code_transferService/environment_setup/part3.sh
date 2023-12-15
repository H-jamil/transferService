#!/bin/bash




cd transferService/final_code_transferService/

echo "Installing Python dependencies..."
pip install -r requirements.txt


# Task 1: Mount tmpfs
echo "Mounting tmpfs to /var/www/html..."
sudo mount -t tmpfs -o size=50G tmpfs /var/www/html

# Task 2: Edit Apache Configuration
echo "Editing Apache configuration file..."
APACHE_CONF="/etc/apache2/apache2.conf"
APACHE_MPM_CONF="
# event MPM
# StartServers: initial number of server processes to start
# MinSpareThreads: minimum number of worker threads which are kept spare
# MaxSpareThreads: maximum number of worker threads which are kept spare
# ThreadsPerChild: constant number of worker threads in each server process
# MaxRequestWorkers: maximum number of worker threads
# MaxConnectionsPerChild: maximum number of requests a server process serves
<IfModule mpm_event_module>
    ServerLimit               40
    StartServers              2
    MinSpareThreads           25
    MaxSpareThreads           75
    ThreadLimit               64
    ThreadsPerChild           25
    MaxRequestWorkers         1000
    MaxClients                1000
    MaxConnectionsPerChild    0
</IfModule>
"
echo "$APACHE_MPM_CONF" | sudo tee -a "$APACHE_CONF"

echo "Restarting Apache..."
sudo apachectl -k restart

# Task 3: Turn off server firewall if on
echo "Checking firewall status..."
sudo ufw status
echo "Disabling firewall..."
sudo ufw disable



echo "Script execution completed."

