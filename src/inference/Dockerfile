FROM public.ecr.aws/lambda/python:3.11

# Copiar dependencias e instalar
COPY src/inference/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copiar código fuente
COPY src/inference/*.py ${LAMBDA_TASK_ROOT}/

# Handler para Lambda
CMD ["handler.lambda_handler"]
