# Server 1: ssh exx or ali@10.0.0.141 (73.241.219.110)
# Server 2: ssh exx@10.0.0.94
# in /etc/ssh/sshd_config, change PasswordAuthentication to no
# change name with: hostnamectl set-hostname <new_name>
echo "creating profile for $1"
echo "and adding public key to authorized_keys"
# add $1 as user to ubuntu server
useradd -m $1
# add public key to authorized_keys
mkdir /home/$1/.ssh
chmod 700 /home/$1/.ssh
touch /home/$1/.ssh/authorized_keys
chmod 600 /home/$1/.ssh/authorized_keys
echo "$2" >> /home/$1/.ssh/authorized_keys
chown -R $1:$1 /home/$1/.ssh
chmod 700 /home/$1/.ssh/authorized_keys
chmod 600 /home/$1/.ssh/authorized_keys
echo "done"

# make bash default shell for $1
chsh -s /bin/bash $1
echo "bash is default shell for $1"

# ask if you want to add $1 to sudoers
echo "Do you want to add $1 to sudoers?"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) echo "Adding $1 to sudoers"; sudo adduser $1 sudo; break;;
        No ) echo "Not adding $1 to sudoers"; break;;
    esac
done

# set password for $1
echo "Setting password for $1"
passwd $1
echo "done"