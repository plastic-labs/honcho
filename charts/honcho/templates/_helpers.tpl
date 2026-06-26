{{- define "honcho.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "honcho.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "honcho.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "honcho.labels" -}}
helm.sh/chart: {{ include "honcho.chart" . }}
{{ include "honcho.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: honcho
{{- end }}

{{- define "honcho.selectorLabels" -}}
app.kubernetes.io/name: {{ include "honcho.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "honcho.componentLabels" -}}
{{ include "honcho.selectorLabels" . }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{- define "honcho.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "honcho.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{- define "honcho.runtimeSecretName" -}}
{{- default (printf "%s-runtime" (include "honcho.fullname" .)) .Values.runtimeSecret.name }}
{{- end }}

{{- define "honcho.dbSecretName" -}}
{{- default (printf "%s-db-credentials" (include "honcho.fullname" .)) .Values.cnpg.credentialsSecret.name }}
{{- end }}

{{- define "honcho.redisName" -}}
{{- default (printf "%s-redis" (include "honcho.fullname" .)) .Values.redis.fullnameOverride }}
{{- end }}

{{- define "honcho.dbName" -}}
{{- default (printf "%s-db" (include "honcho.fullname" .)) .Values.cnpg.fullnameOverride }}
{{- end }}

{{- define "honcho.dbHost" -}}
{{- if .Values.database.host }}
{{- .Values.database.host }}
{{- else if .Values.cnpg.enabled }}
{{- printf "%s-rw.%s.svc.cluster.local" (include "honcho.dbName" .) .Release.Namespace }}
{{- else }}
{{- required "database.host is required when cnpg.enabled=false" .Values.database.host }}
{{- end }}
{{- end }}

{{- define "honcho.waitContainers" -}}
{{- if .Values.waitForDependencies.enabled }}
initContainers:
  {{- if or .Values.cnpg.enabled .Values.database.host }}
  - name: wait-db
    image: {{ .Values.waitForDependencies.image | quote }}
    imagePullPolicy: IfNotPresent
    command:
      - sh
      - -ec
      - |
        until nc -z {{ include "honcho.dbHost" . }} {{ .Values.database.port }}; do
          echo "waiting for postgres"
          sleep 5
        done
  {{- end }}
  {{- if .Values.redis.enabled }}
  - name: wait-redis
    image: {{ .Values.waitForDependencies.image | quote }}
    imagePullPolicy: IfNotPresent
    command:
      - sh
      - -ec
      - |
        until nc -z {{ include "honcho.redisName" . }} {{ .Values.redis.service.port }}; do
          echo "waiting for redis"
          sleep 5
        done
  {{- end }}
{{- end }}
{{- end }}

{{- define "honcho.envFrom" -}}
envFrom:
  - configMapRef:
      name: {{ include "honcho.fullname" . }}-config
  {{- if .Values.runtimeSecret.enabled }}
  - secretRef:
      name: {{ include "honcho.runtimeSecretName" . }}
  {{- end }}
  {{- range .Values.extraEnvFrom }}
  - {{ toYaml . | nindent 4 | trim }}
  {{- end }}
{{- end }}
