use anyhow::{Context, Result};
use rusqlite::Connection;
use std::path::Path;
use std::sync::Mutex;

/// A record for a single transcription request.
pub struct TranscriptionRecord {
    pub id: String,
    pub created_at: String,
    pub audio_filename: Option<String>,
    pub audio_backup_path: Option<String>,
    pub language: String,
    pub transcription: String,
    pub duration_ms: i64,
}

/// SQLite-backed transcription record store.
pub struct TranscriptionDb {
    conn: Mutex<Connection>,
}

impl TranscriptionDb {
    /// Open (or create) the database at `path` and ensure the schema exists.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)
            .with_context(|| format!("Failed to open database at {:?}", path))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS transcription_records (
                id              TEXT PRIMARY KEY,
                created_at      TEXT NOT NULL,
                audio_filename  TEXT,
                audio_backup_path TEXT,
                language        TEXT,
                transcription   TEXT NOT NULL,
                duration_ms     INTEGER NOT NULL
            );",
        )
        .context("Failed to create transcription_records table")?;

        tracing::info!("Database opened at {:?}", path);
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Insert a transcription record.
    pub fn insert_record(&self, record: &TranscriptionRecord) -> Result<()> {
        let conn = self.conn.lock().map_err(|e| anyhow::anyhow!("db lock poisoned: {}", e))?;
        conn.execute(
            "INSERT INTO transcription_records (id, created_at, audio_filename, audio_backup_path, language, transcription, duration_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                record.id,
                record.created_at,
                record.audio_filename,
                record.audio_backup_path,
                record.language,
                record.transcription,
                record.duration_ms,
            ],
        )
        .context("Failed to insert transcription record")?;
        Ok(())
    }
}
