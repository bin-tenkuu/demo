package demo.auth.repository;

import demo.auth.entity.SysUser;
import jakarta.transaction.Transactional;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.jpa.repository.support.JpaRepositoryImplementation;

import java.time.LocalDateTime;
import java.util.Collection;
import java.util.List;

/**
 * @author bin
 * @since 2025/07/15
 */
public interface SysUserRepository extends JpaRepositoryImplementation<SysUser, Long> {

    @Query("update SysUser s set s.status = 1, s.updateBy = :updateBy where s.id in :ids")
    @Modifying
    void updateAllByIdIn(String updateBy, Collection<Long> ids);

    List<Long> findIdByStatusAndIdIn(Integer status, Collection<Long> ids);

    @Query("update SysUser s set s.loginDate = :time where s.id = :id")
    @Modifying
    @Transactional
    void updateLoginById(LocalDateTime time,Long id);
}
