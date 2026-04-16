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

# Lightsail instance is managed manually — not via Terraform.

# -----------------------------------------------------------------------------
# S3 bucket for data storage
# -----------------------------------------------------------------------------

resource "aws_s3_bucket" "data" {
  count  = var.enable_s3_backup ? 1 : 0
  bucket = var.s3_bucket_name

  tags = {
    Project = var.project_name
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  count  = var.enable_s3_backup ? 1 : 0
  bucket = aws_s3_bucket.data[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  count  = var.enable_s3_backup ? 1 : 0
  bucket = aws_s3_bucket.data[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# -----------------------------------------------------------------------------
# IAM user for S3 access (Lightsail has no instance profiles)
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
  name  = "${var.project_name}-s3-access"
  user  = aws_iam_user.backup[0].name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket",
        ]
        Resource = [
          aws_s3_bucket.data[0].arn,
          "${aws_s3_bucket.data[0].arn}/*",
        ]
      }
    ]
  })
}

resource "aws_iam_access_key" "backup" {
  count = var.enable_s3_backup ? 1 : 0
  user  = aws_iam_user.backup[0].name
}
