# Optional Linux build; native Windows is the verified launch path.
FROM python:3.13-slim
WORKDIR /app
COPY vendor/portfolio_llm_evals-0.4.0-py3-none-any.whl /tmp/evals.whl
RUN python -c "import hashlib; assert hashlib.sha256(open('/tmp/evals.whl','rb').read()).hexdigest() == '3f0f45d9c16bf8f7af773cd19bd0425fc2c77b61a74f86ef4f9c7ffbf2bd3b30'"
RUN mv /tmp/evals.whl /tmp/portfolio_llm_evals-0.4.0-py3-none-any.whl && pip install --no-cache-dir /tmp/portfolio_llm_evals-0.4.0-py3-none-any.whl
COPY . /app
RUN pip install --no-cache-dir .
ENV PORTFOLIO_ARTIFACTS=/app/artifacts
ENV OLLAMA_URL=http://host.docker.internal:11434
CMD ["uvicorn", "sql_agent.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8102"]
