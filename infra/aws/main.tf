terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# --- SSH Key Pair ---
resource "tls_private_key" "deploy" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "deploy" {
  key_name   = "${var.project_name}-deploy-key"
  public_key = tls_private_key.deploy.public_key_openssh
}

resource "local_file" "private_key" {
  content         = tls_private_key.deploy.private_key_pem
  filename        = "${path.module}/deploy-key.pem"
  file_permission = "0400"
}

# --- VPC & Networking ---
data "aws_vpc" "default" {
  default = true
}

resource "aws_security_group" "ots" {
  name        = "${var.project_name}-sg"
  description = "Allow SSH + HTTP/HTTPS (Caddy fronts OTS on 80/443)"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.ssh_allowed_cidrs
  }

  ingress {
    description = "HTTP (ACME HTTP-01 challenge + redirect to HTTPS)"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
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
    Name    = "${var.project_name}-sg"
    Project = var.project_name
  }
}

# --- Ubuntu 22.04 LTS AMI ---
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# --- EC2 Instance ---
resource "aws_instance" "ots" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.deploy.key_name
  vpc_security_group_ids = [aws_security_group.ots.id]

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
    encrypted   = true
  }

  user_data = templatefile("${path.module}/user-data.sh", {
    docker_image = var.docker_image
    domain       = var.domain
    tls_email    = var.tls_email
  })

  tags = {
    Name    = "${var.project_name}-server"
    Project = var.project_name
  }
}

# --- Elastic IP ---
resource "aws_eip" "ots" {
  instance = aws_instance.ots.id
  domain   = "vpc"

  tags = {
    Name    = "${var.project_name}-eip"
    Project = var.project_name
  }
}
