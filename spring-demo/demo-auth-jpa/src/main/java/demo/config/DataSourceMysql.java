package demo.config;

import com.mysql.cj.jdbc.MysqlDataSource;
import jakarta.persistence.EntityManagerFactory;
import lombok.RequiredArgsConstructor;
import org.hibernate.boot.archive.scan.internal.DisabledScanner;
import org.hibernate.boot.model.naming.CamelCaseToUnderscoresNamingStrategy;
import org.hibernate.cfg.MappingSettings;
import org.hibernate.cfg.PersistenceSettings;
import org.hibernate.cfg.SchemaToolingSettings;
import org.hibernate.cfg.TransactionSettings;
import org.hibernate.community.dialect.SQLiteDialect;
import org.hibernate.engine.transaction.jta.platform.internal.NoJtaPlatform;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.orm.jpa.EntityManagerFactoryBuilderCustomizer;
import org.springframework.boot.autoconfigure.orm.jpa.HibernateProperties;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.orm.jpa.EntityManagerFactoryBuilder;
import org.springframework.boot.orm.jpa.hibernate.SpringImplicitNamingStrategy;
import org.springframework.context.annotation.Bean;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;
import org.springframework.orm.jpa.JpaTransactionManager;
import org.springframework.orm.jpa.LocalContainerEntityManagerFactoryBean;
import org.springframework.orm.jpa.persistenceunit.PersistenceUnitManager;
import org.springframework.orm.jpa.vendor.HibernateJpaVendorAdapter;
import org.springframework.stereotype.Component;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.annotation.EnableTransactionManagement;

import javax.sql.DataSource;
import java.util.HashMap;
import java.util.Map;

/**
 * @author bin
 * @since 2025/12/15
 */
@Component
@RequiredArgsConstructor
@EnableTransactionManagement
@EnableJpaRepositories(
        entityManagerFactoryRef = "mysqlEntityManagerFactory",
        transactionManagerRef = "mysqlTransactionManager",
        basePackages = "demo.repository"
)
public class DataSourceMysql {

    @Bean("mysqlDataSource")
    @ConfigurationProperties("spring.datasource.mysql")
    public MysqlDataSource dataSource() {
        return new MysqlDataSource();
    }

    @Bean("mysqlEntityManagerFactory")
    public LocalContainerEntityManagerFactoryBean entityManagerFactoryBuilder(
            ObjectProvider<PersistenceUnitManager> persistenceUnitManager,
            ObjectProvider<EntityManagerFactoryBuilderCustomizer> customizers,
            @Qualifier("mysqlDataSource") DataSource dataSource
    ) {
        var jpaVendorAdapter = new HibernateJpaVendorAdapter();
        jpaVendorAdapter.setShowSql(true);
        jpaVendorAdapter.setDatabasePlatform(SQLiteDialect.class.getName());
        jpaVendorAdapter.setGenerateDdl(false);

        var builder = new EntityManagerFactoryBuilder(
                jpaVendorAdapter,
                this::buildJpaProperties,
                persistenceUnitManager.getIfAvailable()
        );
        customizers.orderedStream().forEach((customizer) -> customizer.customize(builder));
        return builder
                .dataSource(dataSource)
                .packages("demo.entity")
                .persistenceUnit("mysql")
                .build();
    }

    /// @see HibernateProperties#determineHibernateProperties
    private Map<String, ?> buildJpaProperties(DataSource dataSource) {
        var map = new HashMap<String, Object>();
        map.put(TransactionSettings.JTA_PLATFORM, new NoJtaPlatform());
        map.put(PersistenceSettings.SCANNER, DisabledScanner.class.getName());
        map.put(MappingSettings.IMPLICIT_NAMING_STRATEGY, SpringImplicitNamingStrategy.class.getName());
        map.put(MappingSettings.PHYSICAL_NAMING_STRATEGY, CamelCaseToUnderscoresNamingStrategy.class.getName());
        map.remove(SchemaToolingSettings.HBM2DDL_AUTO);
        return map;
    }

    @Bean(name = "mysqlTransactionManager")
    public PlatformTransactionManager transactionManager(
            @Qualifier("mysqlEntityManagerFactory") EntityManagerFactory customerEntityManagerFactory
    ) {
        return new JpaTransactionManager(customerEntityManagerFactory);
    }

}
