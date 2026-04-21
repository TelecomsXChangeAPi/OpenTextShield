variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ca-central-1"
}

variable "project_name" {
  description = "Project name used for resource tagging"
  type        = string
  default     = "ots"
}

variable "instance_type" {
  description = "EC2 instance type (t3.medium is the practical minimum for mBERT CPU inference)"
  type        = string
  default     = "t3.medium"
}

variable "docker_image" {
  description = "OpenTextShield Docker image to run"
  type        = string
  default     = "telecomsxchange/opentextshield:latest"
}

variable "ssh_allowed_cidrs" {
  description = "CIDRs allowed to SSH. Restrict to your IP for production (curl ifconfig.me)."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "domain" {
  description = "FQDN the deployment will serve from. DNS A record for this must point to the EIP before first HTTPS request so Caddy can get a Let's Encrypt cert."
  type        = string
  default     = "ots.telecomsxchange.com"
}

variable "tls_email" {
  description = "Email Let's Encrypt uses for expiry notifications + ToS agreement."
  type        = string
  default     = "admin@telecomsxchange.com"
}

variable "root_volume_gb" {
  description = "Root EBS volume size in GB"
  type        = number
  default     = 30
}
