variable "aws_region" {
  description = "AWS region for the Lightsail instance"
  type        = string
  default     = "us-west-2"
}

variable "bundle_id" {
  description = "Lightsail bundle (small_3_0 = $10/mo, 2 GB RAM, 1 vCPU, 60 GB SSD)"
  type        = string
  default     = "small_3_0"
}

variable "blueprint_id" {
  description = "Lightsail OS blueprint"
  type        = string
  default     = "amazon_linux_2023"
}

variable "key_pair_name" {
  description = "Name of the Lightsail key pair for SSH access"
  type        = string
}

variable "project_name" {
  description = "Name prefix for all resources"
  type        = string
  default     = "polymarket-collector"
}

variable "repo_url" {
  description = "Git repository URL to clone"
  type        = string
  default     = "https://github.com/krisseo0202/polymarket_trading.git"
}

variable "repo_branch" {
  description = "Git branch to check out"
  type        = string
  default     = "master"
}

variable "enable_s3_backup" {
  description = "Create an IAM user scoped to the S3 data bucket"
  type        = bool
  default     = true
}

variable "s3_bucket_name" {
  description = "Name of the existing S3 bucket used for data storage"
  type        = string
  default     = "k-polymarket-data"
}
