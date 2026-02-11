package demo.auth.entity;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.Column;
import jakarta.persistence.EntityListeners;
import jakarta.persistence.MappedSuperclass;
import lombok.Getter;
import lombok.Setter;
import org.springframework.data.annotation.CreatedBy;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.LastModifiedBy;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import java.time.LocalDateTime;

/**
 * @author bin
 * @since 2025/12/15
 */
@Getter
@Setter
@MappedSuperclass
@EntityListeners({AuditingEntityListener.class})
public abstract class BaseSys {
    @Schema(description = "创建者")
    @Column(name = "create_by")
    @CreatedBy
    private String createBy;
    @Schema(description = "创建时间")
    @Column(name = "create_time")
    @CreatedDate
    private LocalDateTime createTime;
    @Schema(description = "更新者")
    @Column(name = "update_by")
    @LastModifiedBy
    private String updateBy;
    @Schema(description = "更新时间")
    @Column(name = "update_time")
    @LastModifiedDate
    private LocalDateTime updateTime;
}
