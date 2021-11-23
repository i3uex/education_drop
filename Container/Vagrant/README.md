# Vagrant Container

The contents of this folder are designed to aid as much as possible in the creation of a Vagrant VirtualBox machine based in Ubuntu. To start it, execute `vagrant up` in a terminal window opened in the same folder the file **Vagrantfile** is.

Place your private key **id_rsa** and the SSH configuration file **config** needed to work with the Git repositories in the folder **education_drop/Container/Vagrant/roles/base/tasks**.

Default user is named **vagrant**. Its password is `vagrant`.

Install [VirtualBox Guest Additions][adds] if you want to use different video configurations and shared folders.

[KNIME][knime] has to be installed manually.

All the data needed for this project is in the folder **drop** on the **vagrant** home folder.

[adds]: https://www.itzgeek.com/post/how-to-install-virtualbox-guest-additions-on-ubuntu-20-04/ "How To Install VirtualBox Guest Additions On Ubuntu 20.04"
[knime]: https://www.knime.com "End to end data science for better decision making"
