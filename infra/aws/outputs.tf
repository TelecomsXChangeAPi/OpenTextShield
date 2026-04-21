output "server_ip" {
  description = "Elastic IP of the OTS server — point your DNS A record here"
  value       = aws_eip.ots.public_ip
}

output "ssh_command" {
  description = "SSH command to connect"
  value       = "ssh -i infra/aws/deploy-key.pem ubuntu@${aws_eip.ots.public_ip}"
}

output "api_predict_endpoint" {
  description = "Programmable inference endpoint (POST JSON here)"
  value       = "http://${aws_eip.ots.public_ip}:8002/predict/"
}

output "api_docs" {
  description = "Swagger UI"
  value       = "http://${aws_eip.ots.public_ip}:8002/docs"
}

output "tmf_endpoint" {
  description = "TMForum TMF922 async inference endpoint"
  value       = "http://${aws_eip.ots.public_ip}:8002/tmf-api/aiInferenceJob"
}

output "frontend_url" {
  description = "OTS research frontend"
  value       = "http://${aws_eip.ots.public_ip}:8080"
}

output "health_url" {
  description = "Health check"
  value       = "http://${aws_eip.ots.public_ip}:8002/health"
}

output "metrics_url" {
  description = "Prometheus metrics"
  value       = "http://${aws_eip.ots.public_ip}:8002/metrics"
}

output "private_key_path" {
  description = "Path to the generated SSH private key"
  value       = local_file.private_key.filename
}
