module.exports = {
  apps: [
    {
      name: 'deepseek-okx-bot',
      script: 'deepseek_ok_带市场情绪+指标版本.py',
      interpreter: 'python', // or 'python3' depending on your env
      interpreter_args: '-u', // unbuffered output (equivalent to `python -u`)
      cwd: '.',
      // Logging (pm2 manages logs; no need for nohup)
      out_file: 'plus.out.log',
      error_file: 'plus.err.log',
      merge_logs: true,
      time: true,
      // Process behavior
      exec_mode: 'fork',
      instances: 1,
      watch: false,
      autorestart: true,
      max_restarts: 10,
      restart_delay: 3000,
      env: {
        PRINT_PROMPT: 'True',
        REQUIRE_HIGH_CONFIDENCE_ENTRY: 'True',
        // You can also set RECENT_KLINE_COUNT here, e.g. '20'
        // RECENT_KLINE_COUNT: '20'
      }
    },
    {
      name: 'plus-log-server',
      script: 'plus_log_server.py',
      interpreter: 'python',
      interpreter_args: '-u',
      cwd: '.',
      // Match nohup redirection: combine stdout/stderr to single file
      out_file: 'logserver.log',
      error_file: 'logserver.log',
      merge_logs: true,
      time: true,
      exec_mode: 'fork',
      instances: 1,
      watch: false,
      autorestart: true,
      max_restarts: 10,
      restart_delay: 3000,
      env: {
        PLUS_LOG_PATH: 'plus.out.log',
      }
    }
  ]
}
