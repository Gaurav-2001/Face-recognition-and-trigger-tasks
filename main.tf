provider "aws" {
    region = "ap-south-1"
    profile = "gaurav"        
}

resource "aws_security_group" "allow_http_ssh" {
  name        = "allow_http_ssh"
  description = "Allow HTTP and SSH inbound traffic"

  ingress {
    description = "Allow Traffic from port 80"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    description = "Allow Traffic from port 22"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "allow_80_n_22"
  }
}

resource "aws_instance" "os1" {
    ami = "ami-06a0b4e3b7eb7a300"
    instance_type = "t2.micro"
    tags = { 
        Name = "Instance by TF"
    }
    security_groups = [aws_security_group.allow_http_ssh.name,]
    key_name = "NewAWS_CommonKey"
}