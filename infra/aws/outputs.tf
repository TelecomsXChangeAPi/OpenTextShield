output "server_ip" {
  description = "Elastic IP of the OTS server — point your DNS A record here"
  value       = aws_eip.ots.public_ip
}

output "ssh_command" {
  description = "SSH command to connect"
  value       = "ssh -i infra/aws/deploy-key.pem ubuntu@${aws_eip.ots.public_ip}"
}

output "dns_record_instructions" {
  description = "Create this DNS A record to activate TLS"
  value       = "A  ${var.domain}.  ${aws_eip.ots.public_ip}  TTL 300"
}

output "api_predict_endpoint" {
  description = "Programmable inference endpoint (POST JSON here)"
  value       = "https://${var.domain}/predict/"
}

output "api_docs" {
  description = "Swagger UI"
  value       = "https://${var.domain}/docs"
}

output "tmf_endpoint" {
  description = "TMForum TMF922 async inference endpoint"
  value       = "https://${var.domain}/tmf-api/aiInferenceJob"
}

output "frontend_url" {
  description = "OTS research frontend"
  value       = "https://${var.domain}"
}

output "health_url" {
  description = "Health check"
  value       = "https://${var.domain}/health"
}

output "metrics_url" {
  description = "Prometheus metrics"
  value       = "https://${var.domain}/metrics"
}

output "private_key_path" {
  description = "Path to the generated SSH private key"
  value       = local_file.private_key.filename
}
