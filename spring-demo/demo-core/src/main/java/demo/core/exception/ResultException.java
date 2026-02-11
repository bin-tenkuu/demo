package demo.core.exception;

import lombok.Getter;
import lombok.Setter;

/// @author bin
/// @since 2025/12/15
@Getter
@Setter
public class ResultException extends RuntimeException {
    private int code;

    public ResultException(int code, String msg, Exception e) {
        super(msg, e);
        this.code = code;
    }

    public ResultException(int code, String msg) {
        super(msg);
        this.code = code;
    }

    public ResultException(String msg, Exception e) {
        super(msg, e);
        this.code = 1;
    }

    public ResultException(String msg) {
        super(msg);
        this.code = 1;
    }
}
