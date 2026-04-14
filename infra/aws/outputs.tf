output "public_ip" {
  description = "Static IP address of the Lightsail instance"
  value       = aws_lightsail_static_ip.collector.ip_address
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ~/.ssh/${var.key_pair_name}.pem ec2-user@${aws_lightsail_static_ip.collector.ip_address}"
}

output "s3_bucket" {
  description = "S3 bucket name for data backups"
  value       = var.enable_s3_backup ? aws_s3_bucket.backup[0].bucket : "(disabled)"
}

output "s3_access_key_id" {
  description = "Access key ID for the S3 backup IAM user"
  value       = var.enable_s3_backup ? aws_iam_access_key.backup[0].id : "(disabled)"
  sensitive   = true
}

output "s3_secret_access_key" {
  description = "Secret access key for the S3 backup IAM user"
  value       = var.enable_s3_backup ? aws_iam_access_key.backup[0].secret : "(disabled)"
  sensitive   = true
}

output "post_boot_instructions" {
  description = "Steps to complete after terraform apply"
  value       = <<-EOT

    ===== POST-BOOT SETUP =====

    1. SSH into the instance:
       ssh -i ~/.ssh/${var.key_pair_name}.pem ec2-user@${aws_lightsail_static_ip.collector.ip_address}

    2. Wait for cloud-init to finish (~3-5 min on first boot):
       sudo cloud-init status --wait

    3. Create the .env file:
       sudo -u polymarket tee /opt/polymarket/app/.env << 'EOF'
       HOST=https://clob.polymarket.com
       CHAIN_ID=137
       PRIVATE_KEY=<your-private-key>
       PROXY_FUNDER=<your-proxy-funder>
       API_KEY=<your-api-key>
       API_SECRET=<your-api-secret>
       API_PASSPHRASE=<your-api-passphrase>
       EOF
       sudo chmod 600 /opt/polymarket/app/.env

    4. Start the services:
       sudo systemctl start polymarket-btc-recorder polymarket-snapshots
       sudo systemctl start polymarket-history.timer

    5. Verify:
       sudo systemctl status polymarket-btc-recorder
       sudo systemctl status polymarket-snapshots
       sudo systemctl list-timers polymarket-history
       sudo journalctl -u polymarket-btc-recorder -f

    6. (Optional) Configure S3 backups:
       Run 'terraform output -raw s3_access_key_id' and
       'terraform output -raw s3_secret_access_key' locally,
       then on the instance:
       sudo -u polymarket aws configure
       # Enter the access key, secret key, region=${var.aws_region}

    ============================
  EOT
}
