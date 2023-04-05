# SpeechFirst Recorder

### Introduction

These tools allow you to create an audio dataset using your microphone. Leveraging streamlit to create a web application you can deploy to allow users to submit audio samples.

### Installation

1. Install `ffmpeg` ([Amazon Linux Script](https://gist.github.com/willmasters/382fe6caba44a4345a3de95d98d3aae5))
2. Run:

```yaml
conda env create -f environment.yml
conda activate SpeechFirstRecorder
```

3. Ensure you fully populate the config.yml with your AWS information

### Deploying

#### HTTPS mapping

Due to browser policies, HTTPS must be enabled on the web application to capture microphone input.
We recommend using Amazon Linux instances as it allows you to follow [this tutorial](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/SSL-on-amazon-linux-2.html) to set up HTTPS on your server.

After HTTPS server is set up, you need to map HTTPS port to both Streamlit and to the API ports.
This can be done by mapping:

- `/*` to Streamlit port (`:8501`)
- `/api/*` to API port (`:8000`)

If Apache HTTP Server is being used, add below to `/etc/httpd/conf/httpd.conf` (after `Listen 80` line):

```html
<VirtualHost *:80>
    RewriteEngine on
    RewriteCond %{SERVER_NAME} =speechfirst-ec2-1.appliedinnovationexchange.com
    RewriteRule ^ https://%{SERVER_NAME}%{REQUEST_URI} [END,NE,R=permanent]
</VirtualHost>

<VirtualHost *:443>                                                                                                                    │[ec2-user@ip-172-32-2-129 ~]$ ^C
   ServerName speechfirst-ec2-1.appliedinnovationexchange.com                                                                          │[ec2-user@ip-172-32-2-129 ~]$ sudo service httpd restart
   ProxyPass /api http://localhost:8000                                                                                                │Redirecting to /bin/systemctl restart httpd.service
   ProxyPassReverse /api http://localhost:8000                                                                                         │[ec2-user@ip-172-32-2-129 ~]$ sudo service httpd restart
   RewriteEngine On                                                                                                                    │Redirecting to /bin/systemctl restart httpd.service
   RewriteCond %{HTTP:Upgrade} =websocket                                                                                              │[ec2-user@ip-172-32-2-129 ~]$ sudo service httpd restart
   RewriteCond %{REQUEST_URI} !^/api                                                                                                   │Redirecting to /bin/systemctl restart httpd.service
   RewriteRule /(.*) ws://localhost:8501/$1 [P]                                                                                        │[ec2-user@ip-172-32-2-129 ~]$ sudo service httpd restart
   RewriteCond %{REQUEST_URI} !^/api                                                                                                   │Redirecting to /bin/systemctl restart httpd.service
   RewriteCond %{HTTP:Upgrade} !=websocket                                                                                             │[ec2-user@ip-172-32-2-129 ~]$ sudo service httpd restart
   RewriteRule /(.*) http://localhost:8501/$1 [P]                                                                                      │Redirecting to /bin/systemctl restart httpd.service
   ProxyPass / http://localhost:8501                                                                                                   │[ec2-user@ip-172-32-2-129 ~]$ sudo service httpd restart
   ProxyPassReverse / http://localhost:8501                                                                                            │Redirecting to /bin/systemctl restart httpd.service
   SSLCertificateFile /etc/pki/tls/certs/speechfirst-ec2-1_appliedinnovationexchange_com.crt                                           │[ec2-user@ip-172-32-2-129 ~]$ sudo service httpd restart
   SSLCertificateKeyFile /etc/pki/tls/private/speechfirst-ec2-1_appliedinnovationexchange_com.key                                      │Redirecting to /bin/systemctl restart httpd.service
</VirtualHost>
```

Make sure to adjust `ServerName`, `SSLCertificateFile` & `SSLCertificateKeyFile` values to suit your setup.

Now run `sudo service httpd restart` for changes to take effect.

#### Config file changes

Change the value of `api_url` in `config.yml` to `https://<URL>/api` (replace `<URL>` with your server URL)

### Usage

#### TMUX approach

1. Open tmux with `tmux` (get tmux [here](https://linuxize.com/post/getting-started-with-tmux/))
2. Press `ctrl+B` then `%` to split into two panes
3. Run `python3 -m streamlit run main.py` on the left pane
4. Switch to the right pane by pressing `Ctrl+B` then `right arrow`
5. Run `uvicorn api:app` on the right pane
6. Detach from tmux by pressing `Ctrl + D`, both servers will still run in the background
