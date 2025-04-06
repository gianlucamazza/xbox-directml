# CI/CD per Xbox DirectML App

Questo repository include un workflow GitHub Actions configurato per compilare automaticamente l'applicazione DirectML per Xbox Series S/X.

## Caratteristiche del workflow

- **Compilazione automatica** ad ogni push sul branch main
- **Build ARM64** ottimizzata per Xbox Series S/X
- **Generazione pacchetto MSIX** pronto per il deployment
- **Supporto modelli DirectML** con placeholder automatico
- **Istruzioni di deployment** generate con ogni build

## Come utilizzare il workflow

### Esecuzione manuale

1. Vai alla scheda "Actions" del repository GitHub
2. Seleziona il workflow "Xbox DirectML App Build" dalla lista a sinistra
3. Clicca il pulsante "Run workflow" e conferma l'esecuzione
4. Attendi il completamento della build (circa 5-10 minuti)

### Download degli artefatti

Dopo il completamento della build:

1. Apri l'esecuzione del workflow completata
2. Scorri fino alla sezione "Artifacts" in fondo alla pagina
3. Scarica i seguenti artefatti:
   - **XboxMLApp-Deployment**: Pacchetto ZIP completo con tutti i file necessari
   - **Istruzioni-Deployment**: Guida rapida per il deployment

### Deployment su Xbox

1. Estrai il pacchetto XboxMLApp-Deployment.zip su un PC Windows
2. Segui le istruzioni nel WINDOWS_DEPLOYMENT_GUIDE.md incluso
3. Utilizza Xbox Device Portal (https://[xbox-ip-address]) per caricare il pacchetto MSIX

## Configurazione e personalizzazione

Il workflow è definito nel file `.github/workflows/xbox-directml-build.yml`. Puoi modificarlo per:

- Cambiare le condizioni di trigger (branch o percorsi)
- Aggiungere step di test o validazione
- Configurare la firma del pacchetto con certificati personalizzati
- Modificare le opzioni di build MSBuild

## Requisiti per lo sviluppo locale

Se desideri compilare localmente invece di usare GitHub Actions:

- Windows 10/11
- Visual Studio 2022 con carichi di lavoro UWP e Xbox
- .NET 6.0 SDK
- Windows 10 SDK (10.0.19041.0)

## Problemi noti

- Il certificato generato automaticamente scade dopo 30 giorni
- È necessario abilitare Developer Mode sull'Xbox per l'installazione
- Il modello DirectML deve essere fornito separatamente (vedi MODEL_SETUP_GUIDE.md) 