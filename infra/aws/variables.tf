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

variable "api_allowed_cidrs" {
  description = "CIDRs allowed to hit the OTS API (8002) and frontend (8080)."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "root_volume_gb" {
  description = "Root EBS volume size in GB"
  type        = number
  default     = 30
}
