terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}

# -----------------------------------------------------------------------------
# Lightsail instance
# -----------------------------------------------------------------------------

resource "aws_lightsail_instance" "collector" {
  name              = var.project_name
  availability_zone = "${var.aws_region}a"
  blueprint_id      = var.blueprint_id
  bundle_id         = var.bundle_id
  key_pair_name     = var.key_pair_name
  user_data         = file("${path.module}/user_data.sh")

  tags = {
    Project = var.project_name
  }
}

# -----------------------------------------------------------------------------
# Static IP (free while attached)
# -----------------------------------------------------------------------------

resource "aws_lightsail_static_ip" "collector" {
  name = "${var.project_name}-ip"
}

resource "aws_lightsail_static_ip_attachment" "collector" {
  static_ip_name = aws_lightsail_static_ip.collector.name
  instance_name  = aws_lightsail_instance.collector.name
}

# -----------------------------------------------------------------------------
# Firewall — SSH only
# -----------------------------------------------------------------------------

resource "aws_lightsail_instance_public_ports" "collector" {
  instance_name = aws_lightsail_instance.collector.name

  port_info {
    protocol  = "tcp"
    from_port = 22
    to_port   = 22
  }
}

# -----------------------------------------------------------------------------
# S3 bucket for daily data backups (conditional)
# -----------------------------------------------------------------------------

resource "aws_s3_bucket" "backup" {
  count  = var.enable_s3_backup ? 1 : 0
  bucket = "${var.project_name}-data-${data.aws_caller_identity.current.account_id}"

  tags = {
    Project = var.project_name
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "backup" {
  count  = var.enable_s3_backup ? 1 : 0
  bucket = aws_s3_bucket.backup[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "backup" {
  count  = var.enable_s3_backup ? 1 : 0
  bucket = aws_s3_bucket.backup[0].id

  rule {
    id     = "archive-and-expire"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "GLACIER"
    }

    expiration {
      days = 365
    }
  }
}

resource "aws_s3_bucket_public_access_block" "backup" {
  count  = var.enable_s3_backup ? 1 : 0
  bucket = aws_s3_bucket.backup[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# -----------------------------------------------------------------------------
# IAM user for S3 backups (Lightsail has no instance profiles)
# -----------------------------------------------------------------------------

resource "aws_iam_user" "backup" {
  count = var.enable_s3_backup ? 1 : 0
  name  = "${var.project_name}-backup"

  tags = {
    Project = var.project_name
  }
}

resource "aws_iam_user_policy" "backup" {
  count = var.enable_s3_backup ? 1 : 0
  name  = "${var.project_name}-s3-write"
  user  = aws_iam_user.backup[0].name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:ListBucket",
        ]
        Resource = [
          aws_s3_bucket.backup[0].arn,
          "${aws_s3_bucket.backup[0].arn}/*",
        ]
      }
    ]
  })
}

resource "aws_iam_access_key" "backup" {
  count = var.enable_s3_backup ? 1 : 0
  user  = aws_iam_user.backup[0].name
}
